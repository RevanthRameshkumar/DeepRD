import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add llm_prediction to path to import llm_inference
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
LLM_PREDICTION_DIR = os.path.join(REPO_ROOT, "llm_prediction")
if LLM_PREDICTION_DIR not in sys.path:
    sys.path.insert(0, LLM_PREDICTION_DIR)

from llm_inference import CachingLLM, get_content_from_response


def load_modified_samples(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expecting {"seed":..., "n":..., "samples":[...]}
    if not isinstance(data, dict) or "samples" not in data:
        raise ValueError("Input JSON must contain a 'samples' list.")
    return data


def load_theorem_map() -> Dict[tuple, Dict[str, Any]]:
    """Return a map (source, theorem_id) -> theorem dict including 'title' and 'contents'."""
    # Use relative path to real_world_graphs_and_proofs/data/
    data_dir = os.path.join(os.path.dirname(REPO_ROOT), "real_world_graphs_and_proofs", "data")
    names = [
        ("proofwiki", os.path.join(data_dir, "naturalproofs_proofwiki.json")),
        ("stacks", os.path.join(data_dir, "naturalproofs_stacks.json")),
        ("trench", os.path.join(data_dir, "naturalproofs_trench.json")),
    ]
    mapping = {}
    for source, path in names:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for th in doc.get("dataset", {}).get("theorems", []):
            key = (source, th.get("id"))
            mapping[key] = th
    return mapping


def number_lines(lines: List[str], start: int = 1) -> List[str]:
    return [f"{i}: {line}" for i, line in enumerate(lines, start=start)]


def build_system_prompt(assume_incorrect: bool = False) -> str:
    if assume_incorrect:
        return (
            "You are provided a theorem and a candidate proof for it.\n"
            "Task: There IS a mistake in the proof. Identify the earliest line number where a mistake occurs and explain briefly.\n"
            "Use the line numbers that prefix each proof line.\n"
            "Respond ONLY as strict JSON with keys: \n"
            "  is_correct (must be false),\n"
            "  error_line (integer),\n"
            "  reason (string).\n"
            "Do not include any additional text outside JSON."
        )
    else:
        return (
            "You are provided a theorem and a candidate proof for it.\n"
            "Task: Check whether the proof is correct for the theorem.\n"
            "If the proof is incorrect, identify the earliest line number where a mistake occurs and explain briefly.\n"
            "Use the line numbers that prefix each proof line.\n"
            "Respond ONLY as strict JSON with keys: \n"
            "  is_correct (true/false),\n"
            "  error_line (integer or null),\n"
            "  reason (string).\n"
            "Do not include any additional text outside JSON."
        )


def build_user_message(theorem_title: Optional[str], theorem_contents: Optional[List[str]], numbered_proof: List[str]) -> str:
    title_str = theorem_title or "Untitled Theorem"
    stmt = "\n".join(theorem_contents or [])
    proof_block = "\n".join(numbered_proof)
    return (
        f"Theorem: {title_str}\n\n"
        f"Statement:\n{stmt}\n\n"
        f"Proof (numbered lines):\n{proof_block}\n\n"
        "Return the JSON verdict as instructed."
    )


def build_double_check_prompt() -> str:
    """System prompt for double-checking detected errors in trivial mode."""
    return (
        "You are a mathematical proof verifier reviewing another model's error detection.\n"
        "Task: Determine if the error identified in the proof is a LEGITIMATE mathematical error "
        "or a FALSE POSITIVE (the proof is actually correct at that line).\n\n"
        "You will be given:\n"
        "1. The theorem statement\n"
        "2. The proof (with numbered lines)\n"
        "3. The error line number and reason identified by the first verifier\n\n"
        "Carefully analyze whether the identified error is truly incorrect or if the first verifier made a mistake.\n\n"
        "Respond ONLY as strict JSON with keys:\n"
        "  is_legitimate_error (true/false),\n"
        "  confidence (\"high\"/\"medium\"/\"low\"),\n"
        "  reason (string explaining your assessment).\n"
        "Do not include any additional text outside JSON."
    )


def build_double_check_prompt_perturbed() -> str:
    """System prompt for double-checking detected errors FAR from perturbation in perturbed mode."""
    return (
        "You are reviewing a proof verification task where an INTENTIONAL ERROR was introduced into a proof.\n\n"
        "Context:\n"
        "- An error was intentionally introduced at a specific line (call it Line X)\n"
        "- A verifier model detected a DIFFERENT error at Line Y, which is FAR from Line X\n\n"
        "Task: Determine the nature of the error detected at Line Y:\n"
        "  A. LEGITIMATE ORIGINAL ERROR - An error that existed in the original proof (unrelated to our intentional perturbation)\n"
        "  B. CASCADE EFFECT - A logical consequence or cascading error caused by the perturbation at Line X\n"
        "  C. FALSE POSITIVE - The proof is actually correct at Line Y; the verifier made a mistake\n\n"
        "You will be given:\n"
        "1. The theorem statement\n"
        "2. The proof (with numbered lines)\n"
        "3. The intentional perturbation location and description\n"
        "4. The detected error location and reason\n\n"
        "Respond ONLY as strict JSON with keys:\n"
        "  category (\"A\"/\"B\"/\"C\"),\n"
        "  is_legitimate_original_error (true if category A, false otherwise),\n"
        "  confidence (\"high\"/\"medium\"/\"low\"),\n"
        "  reason (string explaining your assessment).\n"
        "Do not include any additional text outside JSON."
    )


def build_double_check_with_fix_prompt_perturbed() -> str:
    """System prompt for double-checking WITH FIXING - includes request for corrected line."""
    return (
        "You are reviewing a proof verification task where an INTENTIONAL ERROR was introduced into a proof.\n\n"
        "Context:\n"
        "- An error was intentionally introduced at a specific line (call it Line X)\n"
        "- A verifier model detected a DIFFERENT error at Line Y, which is FAR from Line X\n\n"
        "Task: Determine the nature of the error detected at Line Y:\n"
        "  A. LEGITIMATE ORIGINAL ERROR - An error that existed in the original proof (unrelated to our intentional perturbation)\n"
        "  B. CASCADE EFFECT - A logical consequence or cascading error caused by the perturbation at Line X\n"
        "  C. FALSE POSITIVE - The proof is actually correct at Line Y; the verifier made a mistake\n\n"
        "You will be given:\n"
        "1. The theorem statement\n"
        "2. The proof (with numbered lines)\n"
        "3. The intentional perturbation location and description\n"
        "4. The detected error location and reason\n\n"
        "IMPORTANT: If the error is Category A (legitimate original error), you MUST also provide the corrected line.\n\n"
        "Respond ONLY as strict JSON with keys:\n"
        "  category (\"A\"/\"B\"/\"C\"),\n"
        "  is_legitimate_original_error (true if category A, false otherwise),\n"
        "  confidence (\"high\"/\"medium\"/\"low\"),\n"
        "  reason (string explaining your assessment),\n"
        "  corrected_line (string with the fixed version ONLY if category A, otherwise null).\n"
        "Do not include any additional text outside JSON."
    )


def build_double_check_with_fix_prompt_trivial() -> str:
    """System prompt for double-checking WITH FIXING in trivial mode."""
    return (
        "You are a mathematical proof verifier reviewing another model's error detection.\n"
        "Task: Determine if the error identified in the proof is a LEGITIMATE mathematical error "
        "or a FALSE POSITIVE (the proof is actually correct at that line).\n\n"
        "You will be given:\n"
        "1. The theorem statement\n"
        "2. The proof (with numbered lines)\n"
        "3. The error line number and reason identified by the first verifier\n\n"
        "Carefully analyze whether the identified error is truly incorrect or if the first verifier made a mistake.\n\n"
        "IMPORTANT: If this is a legitimate error, you MUST also provide the corrected line.\n\n"
        "Respond ONLY as strict JSON with keys:\n"
        "  is_legitimate_error (true/false),\n"
        "  confidence (\"high\"/\"medium\"/\"low\"),\n"
        "  reason (string explaining your assessment),\n"
        "  corrected_line (string with the fixed version ONLY if legitimate error, otherwise null).\n"
        "Do not include any additional text outside JSON."
    )


def build_double_check_message(
    theorem_title: Optional[str],
    theorem_contents: Optional[List[str]],
    numbered_proof: List[str],
    error_line: int,
    error_reason: str
) -> str:
    """User message for double-checking a detected error."""
    title_str = theorem_title or "Untitled Theorem"
    stmt = "\n".join(theorem_contents or [])
    proof_block = "\n".join(numbered_proof)
    return (
        f"Theorem: {title_str}\n\n"
        f"Statement:\n{stmt}\n\n"
        f"Proof (numbered lines):\n{proof_block}\n\n"
        f"First verifier identified an error:\n"
        f"  - Error line: {error_line}\n"
        f"  - Reason: {error_reason}\n\n"
        "Is this a legitimate mathematical error or a false positive? Return the JSON verdict as instructed."
    )


def build_double_check_message_perturbed(
    theorem_title: Optional[str],
    theorem_contents: Optional[List[str]],
    numbered_proof: List[str],
    perturbation_line: int,
    perturbation_description: str,
    detected_error_line: int,
    detected_error_reason: str
) -> str:
    """User message for double-checking a detected error FAR from the intentional perturbation."""
    title_str = theorem_title or "Untitled Theorem"
    stmt = "\n".join(theorem_contents or [])
    proof_block = "\n".join(numbered_proof)

    return (
        f"Theorem: {title_str}\n\n"
        f"Statement:\n{stmt}\n\n"
        f"Proof (numbered lines):\n{proof_block}\n\n"
        f"INTENTIONAL PERTURBATION:\n"
        f"  - Line: {perturbation_line}\n"
        f"  - Description: {perturbation_description}\n\n"
        f"DETECTED ERROR (far from perturbation):\n"
        f"  - Line: {detected_error_line}\n"
        f"  - Reason: {detected_error_reason}\n\n"
        "Classify this detected error as A (legitimate original error), B (cascade effect), "
        "or C (false positive). Return the JSON verdict as instructed."
    )


def parse_json_response(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try extracting largest JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _to_bool(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "yes"):  # be lenient
            return True
        if v in ("false", "no"):
            return False
    return None


def _to_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int,)):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val.strip())
        except Exception:
            return None
    return None


def compute_metrics(results: List[Dict[str, Any]], gt_incorrect: bool = True, strict: bool = False) -> Dict[str, Any]:
    """Compute accuracy metrics using ground-truth from modified samples.

    Ground truth: every input is intended to be incorrect at exactly the chosen line
    (1-based line number in the numbered proof). We measure:
      - detection_accuracy: predicted is_correct == False rate over all samples.
      - localization_exact_accuracy: among predictions marked incorrect, exact match of error_line.
      - localization_within1_accuracy: same but allowing ±1 tolerance.
      - combined_accuracy: both incorrect detection and exact line match.
      - mae_error_line: mean absolute error for error_line among predictions marked incorrect with a line.
    """
    n = len(results)
    if n == 0:
        return {
            "total": 0,
            "parsed_coverage": 0.0,
            "detection_accuracy": 0.0,
            "localization_exact_accuracy": 0.0,
            "localization_within1_accuracy": 0.0,
            "combined_accuracy": 0.0,
            "mae_error_line": None,
        }

    n_parsed = 0
    n_detect_ok = 0
    n_exact = 0
    n_within1 = 0
    n_combined = 0
    abs_errors: List[int] = []

    # Double-check tracking
    n_double_checked = 0
    n_legitimate_errors = 0
    n_false_positives = 0

    for item in results:
        parsed = item.get("parsed")
        truth_line = None
        try:
            chosen_idx0 = item.get("input", {}).get("chosen_line_index")
            if isinstance(chosen_idx0, int):
                truth_line = chosen_idx0 + 1  # numbered lines start at 1
        except Exception:
            truth_line = None

        if parsed is None:
            continue

        pred_correct = _to_bool(parsed.get("is_correct"))
        pred_line = _to_int(parsed.get("error_line"))

        n_parsed += 1

        # Detection depends on ground-truth setting
        expected = (not gt_incorrect)
        if strict:
            # Only evaluate correctness flag
            if pred_correct is expected:
                n_detect_ok += 1
        else:
            if gt_incorrect:
                if pred_correct is False:
                    n_detect_ok += 1

                    # Check for double-check in perturbed mode
                    dc = item.get("double_check", {})
                    if dc.get("performed") and dc.get("category"):
                        n_double_checked += 1
                        category = dc.get("category")

                        if category == "A":
                            # Legitimate original error - don't penalize for wrong localization
                            # Count as combined success (detected error correctly, location irrelevant)
                            n_legitimate_errors += 1
                            n_within1 += 1
                            n_combined += 1
                        elif category == "B":
                            # Cascade effect - treat as correct localization
                            # This is a valid consequence of the perturbation
                            n_within1 += 1
                            n_combined += 1
                        elif category == "C":
                            # False positive - already not counted, just track
                            n_false_positives += 1
                    elif truth_line is not None and pred_line is not None:
                        # Normal localization check (no double-check performed)
                        if pred_line == truth_line:
                            n_exact += 1
                        if abs(pred_line - truth_line) <= 1:
                            n_within1 += 1
                            n_combined += 1  # both incorrect detection and within-1 localization
                        abs_errors.append(abs(pred_line - truth_line))
            else:
                # Trivial mode: ground truth is that the proof is correct
                if pred_correct is True:
                    n_detect_ok += 1
                    # In trivial mode, localization is irrelevant; treat combined as same as detection
                    n_within1 += 1
                    n_combined += 1
                elif pred_correct is False:
                    # Detected an error in trivial mode - check if double-check was performed
                    dc = item.get("double_check", {})
                    if dc.get("performed"):
                        n_double_checked += 1
                        if dc.get("is_legitimate_error") is True:
                            # This is a real error in the original proof - count as correct detection
                            n_legitimate_errors += 1
                            n_detect_ok += 1
                            n_within1 += 1
                            n_combined += 1
                        elif dc.get("is_legitimate_error") is False:
                            # False positive by the first verifier
                            n_false_positives += 1

    detection_accuracy = n_detect_ok / n if n > 0 else 0.0
    # Localization denominators: predictions marked incorrect with valid line could be fewer than n.
    denom_local = max(1, n_detect_ok)  # avoid div by zero; report 0 if none
    if strict:
        localization_exact_accuracy = None
        localization_within1_accuracy = None
        combined_accuracy = detection_accuracy
        mae_error_line = None
    else:
        localization_exact_accuracy = (n_exact / denom_local) if (n_detect_ok > 0 and gt_incorrect) else (1.0 if n_detect_ok > 0 and not gt_incorrect else 0.0)
        localization_within1_accuracy = (n_within1 / denom_local) if n_detect_ok > 0 else 0.0
        combined_accuracy = (n_combined / n) if n > 0 else 0.0
        mae_error_line = (sum(abs_errors) / len(abs_errors)) if (abs_errors and gt_incorrect) else None

    metrics = {
        "total": n,
        "parsed_coverage": n_parsed / n,
        "detection_accuracy": detection_accuracy,
        "localization_exact_accuracy": localization_exact_accuracy,
        "localization_within1_accuracy": localization_within1_accuracy,
        "combined_accuracy": combined_accuracy,
        "mae_error_line": mae_error_line,
        "counts": {
            "parsed": n_parsed,
            "detect_incorrect": n_detect_ok,
            "exact": n_exact,
            "within1": n_within1,
        },
    }

    # Add double-check statistics if any were performed
    if n_double_checked > 0:
        metrics["double_check_stats"] = {
            "double_checked": n_double_checked,
            "legitimate_errors": n_legitimate_errors,
            "false_positives": n_false_positives,
            "double_check_coverage": n_double_checked / n if n > 0 else 0.0,
        }

    return metrics


def compute_metrics_by_length(
    results: List[Dict[str, Any]],
    bins: int = 10,
    samples_per_bin: int = 10,
    gt_incorrect: bool = True,
    strict: bool = False,
    pre_bins: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Combined accuracy per proof-length bin.

    If samples_per_bin > 0, uses mass-based bins (each bin has ~samples_per_bin items)
    in increasing order of proof length. Otherwise, falls back to equal-width bins
    over the min..max length range.
    """
    # Gather (length, item)
    items: List[Tuple[int, Dict[str, Any]]] = []
    for item in results:
        try:
            L = int(item.get("input", {}).get("proof_length"))
            items.append((L, item))
        except Exception:
            continue

    if not items:
        return {"bins": [], "stats": []}

    def is_combined_hit(it: Dict[str, Any]) -> bool:
        parsed = it.get("parsed")
        if parsed is None:
            return False
        truth_line = None
        try:
            chosen_idx0 = it.get("input", {}).get("chosen_line_index")
            if isinstance(chosen_idx0, int):
                truth_line = chosen_idx0 + 1
        except Exception:
            truth_line = None
        pred_correct = _to_bool(parsed.get("is_correct"))
        pred_line = _to_int(parsed.get("error_line"))
        if strict:
            expected = (not gt_incorrect)
            return pred_correct is expected
        if gt_incorrect:
            if pred_correct is False:
                # Check for double-check in perturbed mode
                dc = it.get("double_check", {})
                if dc.get("performed") and dc.get("category"):
                    category = dc.get("category")
                    if category in ("A", "B"):
                        # A: Legitimate original error, B: Cascade effect - both count as hits
                        return True
                    else:
                        # C: False positive - not a hit
                        return False
                else:
                    # Normal localization check (no double-check performed)
                    return (
                        truth_line is not None
                        and pred_line is not None
                        and abs(pred_line - truth_line) <= 1
                    )
            return False
        else:
            # Trivial mode: consider correct-judgment as the hit
            if pred_correct is True:
                return True
            elif pred_correct is False:
                # Check if double-check confirmed this is a legitimate error
                dc = it.get("double_check", {})
                if dc.get("performed") and dc.get("is_legitimate_error") is True:
                    # Legitimate error - count as hit (no accuracy penalty)
                    return True
            return False

    # Predefined bins provided: assign lengths into these inclusive ranges
    if pre_bins:
        # Normalize bins into list of (lo, hi) ints
        norm_bins: List[Tuple[int, int]] = []
        for rng in pre_bins:
            try:
                lo = int(rng[0]); hi = int(rng[1])
            except Exception:
                continue
            if hi < lo:
                hi = lo
            norm_bins.append((lo, hi))
        if not norm_bins:
            return {"bins": [], "stats": []}

        totals = [0] * len(norm_bins)
        hits = [0] * len(norm_bins)
        for L, it in items:
            bi = None
            for i, (lo, hi) in enumerate(norm_bins):
                if lo <= L <= hi:
                    bi = i; break
            if bi is None:
                continue
            totals[bi] += 1
            if is_combined_hit(it):
                hits[bi] += 1

        stats = []
        for i, (lo, hi) in enumerate(norm_bins):
            tot = totals[i]
            acc = (hits[i] / tot) if tot > 0 else 0.0
            stats.append({
                "bin": i,
                "range": [lo, hi],
                "total": tot,
                "combined_accuracy": acc,
                "hits": hits[i],
            })
        return {"bins": [float(x) for pair in norm_bins for x in pair], "stats": stats}

    if samples_per_bin and samples_per_bin > 0:
        # Mass-based grouping by sorted length
        items.sort(key=lambda t: t[0])
        n = len(items)
        stats = []
        edges_seq: List[float] = []
        i = 0
        b = 0
        while i < n:
            chunk = items[i:i + samples_per_bin]
            i += samples_per_bin
            if not chunk:
                break
            lo_len = float(chunk[0][0])
            hi_len = float(chunk[-1][0])
            total = len(chunk)
            hits = sum(1 for _, it in chunk if is_combined_hit(it))
            stats.append({
                "bin": b,
                "range": [lo_len, hi_len],
                "total": total,
                "combined_accuracy": (hits / total) if total > 0 else 0.0,
                "hits": hits,
            })
            if not edges_seq:
                edges_seq.append(lo_len)
            edges_seq.append(hi_len)
            b += 1
        return {"bins": edges_seq, "stats": stats}

    # Equal-width bins fallback
    lengths = [L for L, _ in items]
    Lmin, Lmax = min(lengths), max(lengths)
    bins_actual = max(1, int(bins))
    if Lmax == Lmin:
        bins_actual = 1
        edges = [float(Lmin), float(Lmax)]
    else:
        width = (Lmax - Lmin) / float(bins_actual)
        edges = [float(Lmin) + i * width for i in range(bins_actual)] + [float(Lmax)]

    def bin_index(L: int) -> int:
        if Lmax == Lmin:
            return 0
        if L == Lmax:
            return bins_actual - 1
        idx = int((L - Lmin) / ((Lmax - Lmin) / float(bins_actual)))
        return max(0, min(bins_actual - 1, idx))

    totals = [0] * bins_actual
    hits = [0] * bins_actual
    for L, it in items:
        b = bin_index(L)
        totals[b] += 1
        if is_combined_hit(it):
            hits[b] += 1

    stats = []
    for i in range(bins_actual):
        lo = edges[i]
        hi = edges[i + 1] if i + 1 < len(edges) else edges[-1]
        tot = totals[i]
        acc = (hits[i] / tot) if tot > 0 else 0.0
        stats.append({
            "bin": i,
            "range": [lo, hi],
            "total": tot,
            "combined_accuracy": acc,
            "hits": hits[i],
        })
    return {"bins": edges, "stats": stats}


def save_accuracy_plot(metrics_by_length: Dict[str, Any], out_path: str) -> Optional[str]:
    """Save a PNG plot of combined accuracy vs. proof length bin midpoints.

    Returns the path to the saved PNG, or None if nothing to plot or if plotting fails.
    """
    stats = metrics_by_length.get("stats", [])
    if not stats:
        return None
    try:
        # Use headless-safe backend
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [(s["range"][0] + s["range"][1]) / 2.0 for s in stats]
        ys = [s.get("combined_accuracy", 0.0) for s in stats]

        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o", linestyle="-", color="#1f77b4")
        plt.xlabel("Proof length")
        plt.ylabel("Combined accuracy (±1)")
        plt.title("Accuracy vs. Proof Length")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.splitext(out_path)[0] + ".png"
        plt.savefig(png_path, dpi=200)
        plt.close()
        return png_path
    except Exception:
        return None

def _filter_results_by_min_length(results: List[Dict[str, Any]], min_length: int) -> List[Dict[str, Any]]:
    if min_length is None or min_length <= 1:
        return results
    filtered: List[Dict[str, Any]] = []
    for it in results:
        try:
            L = int(it.get("input", {}).get("proof_length"))
        except Exception:
            L = None
        if L is None or L >= min_length:
            filtered.append(it)
    return filtered


def run(input_path: str, out_path: str, model: str = "o3-mini", assume_incorrect: bool = False, length_bins: int = 10, min_metric_length: int = 5, trivial: bool = False, strict_metrics: bool = False, pre_bins: Optional[List[List[int]]] = None, double_check: bool = False, double_check_model: str = "o3", double_check_distance_threshold: int = 5, double_check_with_fix: bool = False) -> Dict[str, Any]:
    # Load input samples and theorem map
    payload = load_modified_samples(input_path)
    samples: List[Dict[str, Any]] = payload.get("samples", [])
    th_map = load_theorem_map()

    # Instantiate LLM using project pattern
    system_prompt = build_system_prompt(assume_incorrect=assume_incorrect)
    openai_cache_file = "openai_ai_llm_cache.pkl"

    if '4o' in model:
        params = {"model": model, "temperature": 0.0}
    elif 'DeepSeek-R1' in model or 'deepseek-ai/DeepSeek-R1' in model:
        # R1 needs high max_tokens due to verbose chain-of-thought reasoning
        params = {"model": model, "temperature": 0.0, "max_tokens": 80000}
    else:
        params = {"model": model, "reasoning_effort": "medium"}

    llm = CachingLLM(
        model_params=params,
        initial_messages=[{"role": "system", "content": system_prompt}],
        cache_file=openai_cache_file,
    )

    # Build the per-sample messages
    messages: List[str] = []
    numbered_proofs: List[List[str]] = []
    metadata: List[Dict[str, Any]] = []
    for s in samples:
        key = (s.get("source"), s.get("theorem_id"))
        th = th_map.get(key, {})
        theorem_title = s.get("theorem_title") or th.get("title")
        theorem_contents = th.get("contents") if isinstance(th.get("contents"), list) else []
        if trivial:
            proof_lines = s.get("original_proof") or s.get("modified_proof") or []
        else:
            proof_lines = s.get("modified_proof") or s.get("original_proof") or []
        numbered = number_lines(proof_lines, start=1)
        numbered_proofs.append(numbered)
        messages.append(build_user_message(theorem_title, theorem_contents, numbered))
        metadata.append({
            "source": s.get("source"),
            "theorem_id": s.get("theorem_id"),
            "proof_index": s.get("proof_index"),
        })

    # Query model (batched through CachingLLM)
    responses = llm.generate_responses(messages)

    # Collect outputs
    results: List[Dict[str, Any]] = []
    for s, msg, num_proof, meta, resp in zip(samples, messages, numbered_proofs, metadata, responses):
        resp_dict = None
        content = None
        try:
            resp_dict = resp.to_dict() if resp is not None else None
            content = get_content_from_response(resp_dict) if resp_dict is not None else None
        except Exception:
            content = None
        parsed = parse_json_response(content)

        out_item = {
            "input": {
                "source": s.get("source"),
                "theorem_id": s.get("theorem_id"),
                "theorem_title": s.get("theorem_title"),
                "proof_index": s.get("proof_index"),
                "proof_length": s.get("proof_length"),
                "chosen_line_index": s.get("chosen_line_index"),
                "original_line": s.get("original_line"),
                "modified_line": s.get("modified_line"),
                "numbered_proof": num_proof,
            },
            "raw_system_prompt": system_prompt,
            "raw_user_message": msg,
            "model": model,
            "raw_response": resp_dict,
            "parsed": parsed,
        }
        results.append(out_item)

    # Perform double-check if enabled
    if double_check and trivial:
        print(f"Double-check enabled: verifying detected errors with {double_check_model}...")

        # Identify samples needing double-check (where is_correct=False)
        samples_to_recheck = []
        for idx, result in enumerate(results):
            parsed = result.get("parsed")
            if parsed and _to_bool(parsed.get("is_correct")) is False:
                error_line = _to_int(parsed.get("error_line"))
                error_reason = parsed.get("reason", "")
                if error_line is not None:
                    samples_to_recheck.append((idx, result, error_line, error_reason))

        print(f"Found {len(samples_to_recheck)} samples to double-check...")

        if samples_to_recheck:
            # Setup double-check LLM
            dc_system_prompt = build_double_check_prompt()
            dc_cache_file = "double_check_cache.pkl"

            if '4o' in double_check_model:
                dc_params = {"model": double_check_model, "temperature": 0.0}
            elif 'DeepSeek-R1' in double_check_model or 'deepseek-ai/DeepSeek-R1' in double_check_model:
                dc_params = {"model": double_check_model, "temperature": 0.0, "max_tokens": 80000}
            else:
                dc_params = {"model": double_check_model, "reasoning_effort": "high"}

            dc_llm = CachingLLM(
                model_params=dc_params,
                initial_messages=[{"role": "system", "content": dc_system_prompt}],
                cache_file=dc_cache_file,
            )

            # Build double-check messages
            dc_messages = []
            for idx, result, error_line, error_reason in samples_to_recheck:
                s_input = result["input"]
                key = (s_input.get("source"), s_input.get("theorem_id"))
                th = th_map.get(key, {})
                theorem_contents = th.get("contents") if isinstance(th.get("contents"), list) else []

                dc_msg = build_double_check_message(
                    theorem_title=s_input.get("theorem_title"),
                    theorem_contents=theorem_contents,
                    numbered_proof=s_input.get("numbered_proof", []),
                    error_line=error_line,
                    error_reason=error_reason
                )
                dc_messages.append(dc_msg)

            # Query double-check model
            dc_responses = dc_llm.generate_responses(dc_messages)

            # Update results with double-check data
            for (idx, result, error_line, error_reason), dc_resp in zip(samples_to_recheck, dc_responses):
                dc_resp_dict = None
                dc_content = None
                try:
                    dc_resp_dict = dc_resp.to_dict() if dc_resp is not None else None
                    dc_content = get_content_from_response(dc_resp_dict) if dc_resp_dict is not None else None
                except Exception:
                    dc_content = None

                dc_parsed = parse_json_response(dc_content)

                results[idx]["double_check"] = {
                    "performed": True,
                    "model": double_check_model,
                    "is_legitimate_error": _to_bool(dc_parsed.get("is_legitimate_error")) if dc_parsed else None,
                    "confidence": dc_parsed.get("confidence") if dc_parsed else None,
                    "reason": dc_parsed.get("reason") if dc_parsed else None,
                    "raw_response": dc_resp_dict,
                }

            print(f"Double-check complete for {len(samples_to_recheck)} samples.")

        # Add double_check field to samples that weren't rechecked
        for result in results:
            if "double_check" not in result:
                result["double_check"] = {
                    "performed": False,
                    "model": None,
                    "is_legitimate_error": None,
                    "confidence": None,
                    "reason": None,
                    "raw_response": None,
                }

    # Perturbed mode: double-check errors detected far from perturbation
    if double_check and not trivial:
        print(f"Double-check enabled (perturbed mode): verifying errors far from perturbation with {double_check_model}...")

        # Identify samples where error detected is far from perturbation
        samples_to_recheck = []
        for idx, result in enumerate(results):
            parsed = result.get("parsed")
            if parsed and _to_bool(parsed.get("is_correct")) is False:
                error_line = _to_int(parsed.get("error_line"))
                chosen_line = _to_int(result["input"].get("chosen_line_index"))

                if error_line is not None and chosen_line is not None:
                    distance = abs(error_line - chosen_line)
                    if distance > double_check_distance_threshold:
                        error_reason = parsed.get("reason", "")
                        original_line = result["input"].get("original_line", "")
                        modified_line = result["input"].get("modified_line", "")
                        samples_to_recheck.append((
                            idx, result, error_line, error_reason,
                            chosen_line, original_line, modified_line
                        ))

        print(f"Found {len(samples_to_recheck)} samples with errors >{double_check_distance_threshold} lines from perturbation...")

        if samples_to_recheck:
            # Setup double-check LLM with perturbed mode prompt
            dc_system_prompt = build_double_check_prompt_perturbed()
            dc_cache_file = "double_check_perturbed_cache.pkl"

            if '4o' in double_check_model:
                dc_params = {"model": double_check_model, "temperature": 0.0}
            elif 'DeepSeek-R1' in double_check_model or 'deepseek-ai/DeepSeek-R1' in double_check_model:
                dc_params = {"model": double_check_model, "temperature": 0.0, "max_tokens": 80000}
            else:
                dc_params = {"model": double_check_model, "reasoning_effort": "high"}

            dc_llm = CachingLLM(
                model_params=dc_params,
                initial_messages=[{"role": "system", "content": dc_system_prompt}],
                cache_file=dc_cache_file,
            )

            # Build double-check messages
            dc_messages = []
            for idx, result, error_line, error_reason, chosen_line, original_line, modified_line in samples_to_recheck:
                s_input = result["input"]
                key = (s_input.get("source"), s_input.get("theorem_id"))
                th = th_map.get(key, {})
                theorem_contents = th.get("contents") if isinstance(th.get("contents"), list) else []

                # Build perturbation description
                perturbation_desc = f"Original: '{original_line}' → Modified: '{modified_line}'"

                dc_msg = build_double_check_message_perturbed(
                    theorem_title=s_input.get("theorem_title"),
                    theorem_contents=theorem_contents,
                    numbered_proof=s_input.get("numbered_proof", []),
                    perturbation_line=chosen_line,
                    perturbation_description=perturbation_desc,
                    detected_error_line=error_line,
                    detected_error_reason=error_reason
                )
                dc_messages.append(dc_msg)

            # Query double-check model
            dc_responses = dc_llm.generate_responses(dc_messages)

            # Update results with double-check data
            for (idx, result, error_line, error_reason, chosen_line, original_line, modified_line), dc_resp in zip(samples_to_recheck, dc_responses):
                dc_resp_dict = None
                dc_content = None
                try:
                    dc_resp_dict = dc_resp.to_dict() if dc_resp is not None else None
                    dc_content = get_content_from_response(dc_resp_dict) if dc_resp_dict is not None else None
                except Exception:
                    dc_content = None

                dc_parsed = parse_json_response(dc_content)

                # Extract category (A/B/C)
                category = dc_parsed.get("category") if dc_parsed else None

                results[idx]["double_check"] = {
                    "performed": True,
                    "model": double_check_model,
                    "category": category,
                    "confidence": dc_parsed.get("confidence") if dc_parsed else None,
                    "reason": dc_parsed.get("reason") if dc_parsed else None,
                    "distance_from_perturbation": abs(error_line - chosen_line),
                    "raw_response": dc_resp_dict,
                }

            print(f"Double-check complete for {len(samples_to_recheck)} samples.")

        # Add double_check field to samples that weren't rechecked
        for result in results:
            if "double_check" not in result:
                result["double_check"] = {
                    "performed": False,
                    "model": None,
                    "category": None,
                    "confidence": None,
                    "reason": None,
                    "distance_from_perturbation": None,
                    "raw_response": None,
                }

    # ITERATIVE FIX MODE: Apply fixes for Category A errors and re-verify
    if double_check_with_fix and double_check:
        print(f"\nIterative fix mode enabled: fixing Category A errors and re-verifying...")

        # Setup double-check LLM with FIX prompts
        if trivial:
            dc_fix_system_prompt = build_double_check_with_fix_prompt_trivial()
        else:
            dc_fix_system_prompt = build_double_check_with_fix_prompt_perturbed()

        dc_fix_cache_file = "double_check_fix_cache.pkl"

        if '4o' in double_check_model:
            dc_fix_params = {"model": double_check_model, "temperature": 0.0}
        elif 'DeepSeek-R1' in double_check_model or 'deepseek-ai/DeepSeek-R1' in double_check_model:
            dc_fix_params = {"model": double_check_model, "temperature": 0.0, "max_tokens": 80000}
        else:
            dc_fix_params = {"model": double_check_model, "reasoning_effort": "high"}

        dc_fix_llm = CachingLLM(
            model_params=dc_fix_params,
            initial_messages=[{"role": "system", "content": dc_fix_system_prompt}],
            cache_file=dc_fix_cache_file,
        )

        # Initialize active samples that need fixing
        active_samples = {}
        all_processed_samples = {}  # Track ALL samples including removed ones
        for res_idx, result in enumerate(results):
            # Skip if no double-check was performed
            dc_info = result.get("double_check", {})
            if not dc_info.get("performed"):
                continue

            # Initialize sample state
            sample_data = {
                "result_idx": res_idx,
                "current_proof_lines": result["input"]["numbered_proof"].copy(),
                "chosen_line_idx": result["input"].get("chosen_line_index"),
                "s_input": result["input"],
                "fix_iterations": [],
                "fix_stop_reason": None
            }
            active_samples[res_idx] = sample_data
            all_processed_samples[res_idx] = sample_data  # Keep reference

        total_samples = len(active_samples)
        print(f"Starting iterative fix for {total_samples} samples...")

        # Iterate up to 10 times, processing all active samples in batch
        for iteration in range(10):
            if not active_samples:
                break

            print(f"\n--- Iteration {iteration + 1}/10: Processing {len(active_samples)} active samples ---")

            # BATCH VERIFICATION: Build messages for all active samples
            verify_batch = []
            verify_sample_keys = []
            for sample_key, sample in active_samples.items():
                s_input = sample["s_input"]
                key = (s_input.get("source"), s_input.get("theorem_id"))
                th = th_map.get(key, {})
                theorem_title = s_input.get("theorem_title")
                theorem_contents = th.get("contents") if isinstance(th.get("contents"), list) else []

                verify_msg = build_user_message(theorem_title, theorem_contents, sample["current_proof_lines"])
                verify_batch.append(verify_msg)
                verify_sample_keys.append(sample_key)

            # Execute batch verification
            print(f"  Verifying {len(verify_batch)} samples...")
            verify_responses = llm.generate_responses(verify_batch)

            # Process verification responses
            samples_needing_dc = []
            samples_to_remove = []

            for sample_key, verify_resp in zip(verify_sample_keys, verify_responses):
                sample = active_samples[sample_key]
                verify_content = get_content_from_response(verify_resp.to_dict() if verify_resp else None)
                verify_parsed = parse_json_response(verify_content)

                # Check if proof is now correct
                if verify_parsed and _to_bool(verify_parsed.get("is_correct")) is not False:
                    sample["fix_stop_reason"] = "marked_correct"
                    sample["fix_iterations"].append({
                        "iteration": iteration,
                        "action": "verification",
                        "result": "marked_correct",
                        "parsed": verify_parsed
                    })
                    samples_to_remove.append(sample_key)
                    print(f"    Sample {sample_key}: Stopped - marked_correct")
                    continue

                # Error detected
                error_line = _to_int(verify_parsed.get("error_line")) if verify_parsed else None
                error_reason = verify_parsed.get("reason", "") if verify_parsed else ""

                if error_line is None:
                    sample["fix_stop_reason"] = "no_error_line"
                    sample["fix_iterations"].append({
                        "iteration": iteration,
                        "action": "verification",
                        "result": "no_error_line",
                        "parsed": verify_parsed
                    })
                    samples_to_remove.append(sample_key)
                    print(f"    Sample {sample_key}: Stopped - no_error_line")
                    continue

                # Check if error is at perturbation (within ±1 line)
                if not trivial and sample["chosen_line_idx"] is not None:
                    chosen_line_1based = sample["chosen_line_idx"] + 1
                    if abs(error_line - chosen_line_1based) <= 1:
                        sample["fix_stop_reason"] = "found_perturbation"
                        sample["fix_iterations"].append({
                            "iteration": iteration,
                            "action": "verification",
                            "result": "found_perturbation",
                            "error_line": error_line,
                            "distance": abs(error_line - chosen_line_1based),
                            "parsed": verify_parsed
                        })
                        samples_to_remove.append(sample_key)
                        print(f"    Sample {sample_key}: Stopped - found_perturbation at line {error_line}")
                        continue

                # Need double-check - add to batch
                # IMPORTANT: Store this verification in iterations so we can use it later for metrics!
                sample["verify_parsed"] = verify_parsed
                sample["error_line"] = error_line
                sample["error_reason"] = error_reason
                # Record this verification (will be followed by double-check in same iteration)
                sample["current_verify_iteration"] = {
                    "iteration": iteration,
                    "action": "verification",
                    "error_line": error_line,
                    "parsed": verify_parsed
                }
                samples_needing_dc.append(sample_key)

            # Remove stopped samples
            for sample_key in samples_to_remove:
                del active_samples[sample_key]

            # BATCH DOUBLE-CHECK: Build messages for samples needing double-check
            if samples_needing_dc:
                dc_batch = []
                dc_sample_keys = []

                for sample_key in samples_needing_dc:
                    sample = active_samples[sample_key]
                    s_input = sample["s_input"]
                    key = (s_input.get("source"), s_input.get("theorem_id"))
                    th = th_map.get(key, {})
                    theorem_title = s_input.get("theorem_title")
                    theorem_contents = th.get("contents") if isinstance(th.get("contents"), list) else []

                    if trivial:
                        dc_fix_msg = build_double_check_message(
                            theorem_title=theorem_title,
                            theorem_contents=theorem_contents,
                            numbered_proof=sample["current_proof_lines"],
                            error_line=sample["error_line"],
                            error_reason=sample["error_reason"]
                        )
                    else:
                        perturbation_line = sample["chosen_line_idx"] + 1 if sample["chosen_line_idx"] is not None else -1
                        perturbation_description = s_input.get("modified_line", "")
                        dc_fix_msg = build_double_check_message_perturbed(
                            theorem_title=theorem_title,
                            theorem_contents=theorem_contents,
                            numbered_proof=sample["current_proof_lines"],
                            perturbation_line=perturbation_line,
                            perturbation_description=perturbation_description,
                            detected_error_line=sample["error_line"],
                            detected_error_reason=sample["error_reason"]
                        )

                    dc_batch.append(dc_fix_msg)
                    dc_sample_keys.append(sample_key)

                # Execute batch double-check
                print(f"  Double-checking {len(dc_batch)} samples...")
                dc_responses = dc_fix_llm.generate_responses(dc_batch)

                # Process double-check responses
                samples_to_remove = []
                for sample_key, dc_fix_resp in zip(dc_sample_keys, dc_responses):
                    sample = active_samples[sample_key]
                    dc_fix_content = get_content_from_response(dc_fix_resp.to_dict() if dc_fix_resp else None)
                    dc_fix_parsed = parse_json_response(dc_fix_content)

                    if not dc_fix_parsed:
                        sample["fix_stop_reason"] = "double_check_parse_error"
                        sample["fix_iterations"].append({
                            "iteration": iteration,
                            "action": "double_check",
                            "result": "parse_error",
                            "error_line": sample["error_line"]
                        })
                        samples_to_remove.append(sample_key)
                        print(f"    Sample {sample_key}: Stopped - double_check_parse_error")
                        continue

                    # Add the verification iteration that preceded this double-check
                    # (stored in current_verify_iteration at line 1061)
                    if "current_verify_iteration" in sample:
                        sample["fix_iterations"].append(sample["current_verify_iteration"])
                        del sample["current_verify_iteration"]  # Clean up

                    # Determine category/legitimacy
                    if trivial:
                        is_legitimate = _to_bool(dc_fix_parsed.get("is_legitimate_error"))
                        if is_legitimate is False:
                            sample["fix_stop_reason"] = "false_positive"
                            sample["fix_iterations"].append({
                                "iteration": iteration,
                                "action": "double_check",
                                "result": "false_positive",
                                "error_line": sample["error_line"],
                                "dc_parsed": dc_fix_parsed
                            })
                            samples_to_remove.append(sample_key)
                            print(f"    Sample {sample_key}: Stopped - false_positive")
                            continue
                    else:
                        category = dc_fix_parsed.get("category")
                        if category == "C":
                            sample["fix_stop_reason"] = "false_positive"
                            sample["fix_iterations"].append({
                                "iteration": iteration,
                                "action": "double_check",
                                "result": "false_positive_category_C",
                                "error_line": sample["error_line"],
                                "category": category,
                                "dc_parsed": dc_fix_parsed
                            })
                            samples_to_remove.append(sample_key)
                            print(f"    Sample {sample_key}: Stopped - false_positive (Category C)")
                            continue
                        elif category == "B":
                            sample["fix_stop_reason"] = "cascade_effect"
                            sample["fix_iterations"].append({
                                "iteration": iteration,
                                "action": "double_check",
                                "result": "cascade_effect_category_B",
                                "error_line": sample["error_line"],
                                "category": category,
                                "dc_parsed": dc_fix_parsed
                            })
                            samples_to_remove.append(sample_key)
                            print(f"    Sample {sample_key}: Stopped - cascade_effect (Category B)")
                            continue
                        elif category != "A":
                            sample["fix_stop_reason"] = "unknown_category"
                            sample["fix_iterations"].append({
                                "iteration": iteration,
                                "action": "double_check",
                                "result": "unknown_category",
                                "error_line": sample["error_line"],
                                "category": category,
                                "dc_parsed": dc_fix_parsed
                            })
                            samples_to_remove.append(sample_key)
                            print(f"    Sample {sample_key}: Stopped - unknown_category ({category})")
                            continue

                    # Category A (legitimate error) - FIX IT
                    corrected_line = dc_fix_parsed.get("corrected_line")
                    if not corrected_line:
                        sample["fix_stop_reason"] = "no_corrected_line"
                        sample["fix_iterations"].append({
                            "iteration": iteration,
                            "action": "fix_attempt",
                            "result": "no_corrected_line",
                            "error_line": sample["error_line"],
                            "dc_parsed": dc_fix_parsed
                        })
                        samples_to_remove.append(sample_key)
                        print(f"    Sample {sample_key}: Stopped - no_corrected_line")
                        continue

                    # Apply the fix
                    error_line = sample["error_line"]
                    if 0 < error_line <= len(sample["current_proof_lines"]):
                        old_line = sample["current_proof_lines"][error_line - 1]
                        sample["current_proof_lines"][error_line - 1] = f"{error_line}: {corrected_line}"
                        sample["fix_iterations"].append({
                            "iteration": iteration,
                            "action": "fix_applied",
                            "result": "success",
                            "error_line": error_line,
                            "old_line": old_line,
                            "new_line": sample["current_proof_lines"][error_line - 1],
                            "dc_parsed": dc_fix_parsed
                        })
                        print(f"    Sample {sample_key}: Fixed error at line {error_line}")
                    else:
                        sample["fix_stop_reason"] = "invalid_line_number"
                        sample["fix_iterations"].append({
                            "iteration": iteration,
                            "action": "fix_attempt",
                            "result": "invalid_line_number",
                            "error_line": error_line,
                            "proof_length": len(sample["current_proof_lines"])
                        })
                        samples_to_remove.append(sample_key)
                        print(f"    Sample {sample_key}: Stopped - invalid_line_number")
                        continue

                # Remove stopped samples
                for sample_key in samples_to_remove:
                    del active_samples[sample_key]

        # Mark any remaining samples as hitting max_iterations
        for sample_key, sample in active_samples.items():
            sample["fix_stop_reason"] = "max_iterations"
            print(f"  Sample {sample_key}: Stopped - max_iterations")

        # Store results back to original results array
        for res_idx, result in enumerate(results):
            dc_info = result.get("double_check", {})
            if not dc_info.get("performed"):
                result["fix_with_check"] = {
                    "performed": False,
                    "iterations": [],
                    "final_iteration_count": 0,
                    "stop_reason": None,
                    "final_proof": None
                }
                continue

            # Find sample data from all_processed_samples (includes removed ones)
            if res_idx in all_processed_samples:
                sample = all_processed_samples[res_idx]
            else:
                # Sample wasn't in the fix loop (shouldn't happen)
                result["fix_with_check"] = {
                    "performed": False,
                    "iterations": [],
                    "final_iteration_count": 0,
                    "stop_reason": None,
                    "final_proof": None
                }
                continue

            result["fix_with_check"] = {
                "performed": True,
                "iterations": sample["fix_iterations"],
                "final_iteration_count": len(sample["fix_iterations"]),
                "stop_reason": sample["fix_stop_reason"],
                "final_proof": sample["current_proof_lines"] if sample["fix_iterations"] else None
            }

            # Update the parsed field with the final verification result
            # This is critical for metrics calculation!
            if sample["fix_iterations"]:
                # Find the LAST verification result (iterate backwards)
                for iteration in reversed(sample["fix_iterations"]):
                    if iteration.get("action") == "verification" and "parsed" in iteration:
                        # Update the main parsed field with the final verification
                        result["parsed"] = iteration["parsed"]
                        break

                # CRITICAL: Clear the double_check field for samples that went through fix loop
                # The original double_check is no longer relevant - we have a NEW verification
                # after applying fixes. If we don't clear this, the metrics calculation will
                # incorrectly exclude this sample from MAE/localization metrics.
                if "double_check" in result:
                    result["double_check"]["performed"] = False

        # Store all processed samples (including those removed during iterations)
        # We need to track samples that were removed - let's create a separate structure
        all_processed = {}
        for res_idx, result in enumerate(results):
            if result.get("fix_with_check", {}).get("performed"):
                all_processed[res_idx] = result["fix_with_check"]

        # Update results with processed data
        for res_idx in all_processed:
            results[res_idx]["fix_with_check"] = all_processed[res_idx]

        # Add fix_with_check field to samples that weren't fixed
        for result in results:
            if "fix_with_check" not in result:
                result["fix_with_check"] = {
                    "performed": False,
                    "iterations": [],
                    "final_iteration_count": 0,
                    "stop_reason": None,
                    "final_proof": None
                }

        print(f"Iterative fix complete.")

    # Compute metrics based on outputs (optionally ignoring very short proofs)
    results_for_metrics = _filter_results_by_min_length(results, min_metric_length)
    gt_incorrect = not trivial
    metrics = compute_metrics(results_for_metrics, gt_incorrect=gt_incorrect, strict=strict_metrics)
    metrics_by_length = compute_metrics_by_length(results_for_metrics, bins=length_bins, gt_incorrect=gt_incorrect, strict=strict_metrics, pre_bins=pre_bins)
    accuracy_plot_path = save_accuracy_plot(metrics_by_length, out_path)

    output = {
        "model": model,
        "input_file": input_path,
        "count": len(results),
        "system_prompt": system_prompt,
        "assume_incorrect": assume_incorrect,
        "trivial_mode": trivial,
        "double_check_enabled": double_check,
        "double_check_model": double_check_model if double_check else None,
        "results": results,
        "metrics": metrics,
        "metrics_by_length": metrics_by_length,
        "metrics_min_proof_length": min_metric_length,
        "accuracy_plot": accuracy_plot_path,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def main():
    parser = argparse.ArgumentParser(description="Check proofs with o3-mini; output JSON with verdicts and line numbers.")
    parser.add_argument("--input", type=str, default=os.path.join("wikidata", "modified_samples.json"), help="Input JSON from sample_and_corrupt_proofs.py")
    parser.add_argument("--out", type=str, default="proof_check_results_o3mini.json", help="Output JSON path")

    # parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1", help="Model name (default: o3-mini)")
    parser.add_argument("--model", type=str, default="o3-mini", help="Model name (default: o3-mini)")
    # parser.add_argument("--model", type=str, default="gpt-4o", help="Model name (default: o3-mini)")
    # parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3", help="Model name (default: o3-mini)")

    parser.add_argument("--length-bins", type=int, default=10, help="Number of bins for accuracy-by-length histogram")
    parser.add_argument("--pre-bins", action="store_true", help="Use bin_boundaries_inclusive from the input JSON to define histogram bins (mutually exclusive with --length-bins and samples-per-bin modes)")
    parser.add_argument("--min-metric-length", type=int, default=5, help="Ignore proofs with length < this value when computing metrics (set 1 to include all)")
    parser.add_argument("--metrics-only", type=str, default=None, help="If set, compute metrics for an existing results JSON and print; no inference.")
    parser.add_argument("--assume-incorrect", action="store_true", help="Change system prompt to assert there IS a mistake and the task is to find the earliest erroneous line.")
    parser.add_argument("--trivial", action="store_true", help="Evaluate unmodified proofs; metrics expect is_correct == True and filename gets .trivial appended")
    parser.add_argument("--strict-metrics", action="store_true", help="Only evaluate correctness (is_correct) without checking error_line localization")
    parser.add_argument("--double-check", default=True, action="store_true", help="Use a secondary model to verify detected errors. In trivial mode: checks all flagged errors. In perturbed mode: checks errors far from perturbation")
    parser.add_argument("--double-check-model", type=str, default="o3", help="Model to use for double-checking detected errors (default: o3)")
    parser.add_argument("--double-check-distance-threshold", type=int, default=5, help="In perturbed mode, minimum line distance from perturbation to trigger double-check (default: 5)")
    parser.add_argument("--double-check-with-fix", default=True, action="store_true", help="When double-check finds legitimate errors (Category A), fix them iteratively until perturbation found or no more errors. Requires --double-check.")
    args = parser.parse_args()

    # Validation: --double-check-with-fix requires --double-check
    if args.double_check_with_fix and not args.double_check:
        parser.error("--double-check-with-fix requires --double-check to be enabled")

    if args.metrics_only:
        with open(args.metrics_only, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        pre_bins = None
        if args.pre_bins:
            pre_bins = data.get("bin_boundaries_inclusive")
        results_for_metrics = _filter_results_by_min_length(results, args.min_metric_length)
        gt_incorrect = not args.trivial
        metrics = compute_metrics(results_for_metrics, gt_incorrect=gt_incorrect, strict=args.strict_metrics)
        metrics_by_length = compute_metrics_by_length(results_for_metrics, bins=args.length_bins, gt_incorrect=gt_incorrect, strict=args.strict_metrics, pre_bins=pre_bins)
        print(json.dumps({"metrics": metrics, "metrics_by_length": metrics_by_length}, indent=2))
        return

    out_path = args.out
    if args.trivial:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}.trivial{ext}"

    # If --pre-bins, pull from input JSON
    pre_bins = None
    if args.pre_bins:
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                input_payload = json.load(f)
            pre_bins = input_payload.get("bin_boundaries_inclusive")
        except Exception:
            pre_bins = None

    run(
        args.input,
        out_path,
        model=args.model,
        assume_incorrect=args.assume_incorrect,
        length_bins=args.length_bins,
        min_metric_length=args.min_metric_length,
        trivial=args.trivial,
        strict_metrics=args.strict_metrics,
        pre_bins=pre_bins,
        double_check=args.double_check,
        double_check_model=args.double_check_model,
        double_check_distance_threshold=args.double_check_distance_threshold,
        double_check_with_fix=args.double_check_with_fix,
    )


if __name__ == "__main__":
    main()

    
