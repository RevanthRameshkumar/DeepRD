import argparse
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Add llm_prediction to path to import llm_inference
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
LLM_PREDICTION_DIR = os.path.join(REPO_ROOT, "llm_prediction")
if LLM_PREDICTION_DIR not in sys.path:
    sys.path.insert(0, LLM_PREDICTION_DIR)

from llm_inference import CachingLLM, get_content_from_response  # type: ignore

# Verbose progress logging (always enabled)
_VERBOSE = True


# -------------------------------
# Data structures
# -------------------------------

@dataclass
class ProofRef:
    source: str  # one of {proofwiki, stacks, trench}
    theorem_id: Any
    theorem_title: Optional[str]
    proof_index: int
    contents: List[str]


# -------------------------------
# Utilities
# -------------------------------

TRIVIAL_PREFIXES = (
    "Then:",
    "So:",
    "Thus:",
    "Hence:",
    "Therefore:",
    "Observe that:",
    "We have that:",
    "It follows that:",
)

TRIVIAL_EXACT = {
    "}}",
    "{{qed}}",
}

TRIVIAL_CONTAINS = (
    "{{begin-eqn}}",
    "{{end-eqn}}",
)


def is_substantial_line(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if not s:
        return False
    # If line starts with an enumeration label like :$(1) \quad ..., strip it and keep evaluating the rest
    m = re.match(r"^:?\$\(\d+\)\s*\\quad\s*", s)
    if m:
        s = s[m.end():].strip()
    if s in TRIVIAL_EXACT:
        return False
    for p in TRIVIAL_PREFIXES:
        if s.startswith(p):
            return False
    for c in TRIVIAL_CONTAINS:
        if c in s:
            return False
    # Require at least one alphanumeric or common math symbol
    if not re.search(r"[A-Za-z0-9]", s):
        return False
    # Avoid lines that are too short (likely scaffolding)
    if len(s) < 6:
        return False
    return True


def load_datasets() -> List[ProofRef]:
    # Use relative path to real_world_graphs_and_proofs/data/
    data_dir = os.path.join(os.path.dirname(CURRENT_DIR), "real_world_graphs_and_proofs", "data")
    files = [
        ("proofwiki", os.path.join(data_dir, "naturalproofs_proofwiki.json")),
        ("stacks", os.path.join(data_dir, "naturalproofs_stacks.json")),
        ("trench", os.path.join(data_dir, "naturalproofs_trench.json")),
    ]
    proofs: List[ProofRef] = []
    for source, path in files:
        if not os.path.exists(path):
            # Allow missing optional sources but continue
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        theorems = data.get("dataset", {}).get("theorems", [])
        for th in theorems:
            th_id = th.get("id")
            th_title = th.get("title")
            for pidx, p in enumerate(th.get("proofs", [])):
                contents = p.get("contents", []) or []
                if contents:
                    proofs.append(
                        ProofRef(
                            source=source,
                            theorem_id=th_id,
                            theorem_title=th_title,
                            proof_index=pidx,
                            contents=contents,
                        )
                    )
    return proofs


def compute_bins(lengths: List[int], n_bins: int) -> List[Tuple[int, int]]:
    if not lengths:
        return []
    values = sorted(set(lengths))
    if len(values) <= n_bins:
        # Make each unique length its own bin, merging to reach n_bins
        bins = []
        for i, v in enumerate(values):
            left = v
            right = v
            bins.append((left, right))
        # If fewer unique than bins requested, pad by extending last bin
        while len(bins) < n_bins:
            bins.append((values[-1], values[-1]))
        return bins[:n_bins]

    # Quantile-based bin edges (inclusive ranges)
    # Use ranks rather than numpy to avoid dependency
    sorted_lengths = sorted(lengths)
    bins: List[Tuple[int, int]] = []
    for i in range(n_bins):
        q_lo = i / n_bins
        q_hi = (i + 1) / n_bins
        idx_lo = max(0, min(len(sorted_lengths) - 1, int(math.floor(q_lo * (len(sorted_lengths) - 1)))))
        idx_hi = max(0, min(len(sorted_lengths) - 1, int(math.floor(q_hi * (len(sorted_lengths) - 1)))))
        left = sorted_lengths[idx_lo]
        right = sorted_lengths[idx_hi]
        if bins and left <= bins[-1][1]:
            # Ensure monotonic progress by nudging right edge if needed
            left = bins[-1][1]
        bins.append((left, right))
    # Normalize to non-decreasing, inclusive bins
    fixed: List[Tuple[int, int]] = []
    prev_right = None
    for left, right in bins:
        if prev_right is None:
            fixed.append((left, right))
        else:
            l = max(left, prev_right)
            fixed.append((l, max(l, right)))
        prev_right = fixed[-1][1]
    return fixed


def line_indices_substantial(contents: List[str]) -> List[int]:
    idxs = [i for i, ln in enumerate(contents) if is_substantial_line(ln)]
    return idxs


def pick_line(contents: List[str], rng: random.Random) -> Optional[int]:
    idxs = line_indices_substantial(contents)
    if not idxs:
        return None
    return rng.choice(idxs)




SEMANTIC_TOKENS = set([
    r"\\le", r"\\ge", r"\\lt", r"\\gt", "≤", "≥", "<", ">", "=", r"\\neq",
    r"\\in", r"\\notin", r"\\subset", r"\\subseteq", r"\\supset", r"\\supseteq",
    r"\\iff", r"\\implies", r"\\sum", r"\\prod", r"\\forall", r"\\exists",
])


def _math_segments(s: str) -> List[str]:
    segs = []
    for m in re.finditer(r"\$(.+?)\$", s):
        segs.append(m.group(1))
    for m in re.finditer(r"\\\((.+?)\\\)", s):
        segs.append(m.group(1))
    for m in re.finditer(r"\\\[(.+?)\\\]", s):
        segs.append(m.group(1))
    return segs


def extract_semantic_tokens(text: str) -> set:
    toks = set()
    segments = _math_segments(text)
    scan_targets = segments if segments else [text]
    for part in scan_targets:
        for t in SEMANTIC_TOKENS:
            if t in part:
                toks.add(t)
        if re.search(r"(?<![A-Za-z])\+(?![A-Za-z])", part):
            toks.add("+")
        if re.search(r"(?<![A-Za-z])\-(?![A-Za-z])", part):
            toks.add("-")
    return toks


def numbers_changed(orig: str, new: str) -> bool:
    orig_nums = re.findall(r"\d+", orig)
    new_nums = re.findall(r"\d+", new)
    return orig_nums != new_nums


def is_meaningful_change(orig: str, mod: str) -> bool:
    """Check if modification represents a meaningful mathematical change.

    Returns True if the change is meaningful (should be accepted).
    Returns False if the change is only formatting/whitespace (should be rejected).
    """
    if not mod or mod == orig:
        return False

    # Check if numbers changed
    if numbers_changed(orig, mod):
        return True

    # Check if semantic tokens changed
    if extract_semantic_tokens(orig) != extract_semantic_tokens(mod):
        return True

    # Even if semantic tokens are the same, check if the actual mathematical content changed
    # by looking at variables, function names, and LaTeX commands

    # Extract all LaTeX commands (e.g., \sin, \cos, \alpha, \beta)
    orig_commands = set(re.findall(r'\\[a-zA-Z]+', orig))
    mod_commands = set(re.findall(r'\\[a-zA-Z]+', mod))
    if orig_commands != mod_commands:
        return True

    # Extract all single-letter variables and check for changes
    # (excluding whitespace and common LaTeX markup)
    orig_vars = set(re.findall(r'[a-zA-Z]', re.sub(r'\\[a-zA-Z]+', '', orig)))
    mod_vars = set(re.findall(r'[a-zA-Z]', re.sub(r'\\[a-zA-Z]+', '', mod)))
    if orig_vars != mod_vars:
        return True

    # Check for operator changes not caught by semantic tokens (*, /, ^, etc.)
    orig_ops = set(re.findall(r'[\+\-\*/\^_{}()[\]]', orig))
    mod_ops = set(re.findall(r'[\+\-\*/\^_{}()[\]]', mod))
    if orig_ops != mod_ops:
        return True

    # If we get here, the changes are likely only formatting/whitespace
    return False


def batch_gpt_pick_lines(contents_list: List[List[str]], model: str = "gpt-4o", cache_file: str = "openai_ai_llm_cache_sampler_2.pkl") -> List[Optional[int]]:
    if not contents_list:
        return []
    chosen: List[Optional[int]] = [None] * len(contents_list)
    prompts: List[str] = []
    query_map: List[int] = []  # maps prompt index -> contents_list index
    candidates_list: List[List[int]] = []

    for idx, contents in enumerate(contents_list):
        base_candidates = line_indices_substantial(contents)
        # Prefer lines with clear math signals; fallback to all substantial lines
        def _has_math_signal(s: str) -> bool:
            if re.search(r"\\(le|ge|lt|gt|in|notin|subset|subseteq|supset|supseteq|sum|prod|forall|exists)\\b", s):
                return True
            if any(x in s for x in ["=", "≤", "≥", "<", ">", "⊆", "⊇", "⊂", "⊃"]):
                return True
            if re.search(r"\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", s):
                return True
            return False
        candidates_math = [i for i in base_candidates if _has_math_signal(contents[i])]
        candidates = candidates_math if candidates_math else base_candidates
        candidates_list.append(candidates)
        if not candidates:
            continue
        numbered = [f"{i+1}: {ln}" for i, ln in enumerate(contents)]
        proof_block = "\n".join(numbered)
        allowed_indices = ", ".join(str(i) for i in candidates)
        prompt = (
            "You are given a proof as numbered lines. Choose ONE consequential line (not scaffolding, not labels, "
            "not formatting like {{...}}, not 'Then:'/'So:' style) whose change would invalidate the proof.\n"
            "Allowed indices (0-based): [" + allowed_indices + "]\n"
            "Respond ONLY with the single integer index (0-based). No words, no JSON.\n\n"
            "Proof (numbered):\n" + proof_block
        )
        prompts.append(prompt)
        query_map.append(idx)

    if not prompts:
        return chosen

    resps = get_gpt_pick_llm(model, cache_file=cache_file).generate_responses(prompts)

    for ci, r in zip(query_map, resps):
        contents = contents_list[ci]
        candidates = candidates_list[ci]
        try:
            out = get_content_from_response(r.to_dict())
        except Exception:
            out = None
        if not out:
            continue
        # Try strict integer, else try JSON, else first integer in text
        idx_val = None
        s = out.strip()
        if re.fullmatch(r"[-+]?\d+", s):
            try:
                idx_val = int(s)
            except Exception:
                idx_val = None
        if idx_val is None:
            try:
                data = json.loads(out)
                if isinstance(data, dict):
                    cand = data.get("line_index") or data.get("index") or data.get("line_number")
                    if cand is not None:
                        idx_val = int(cand)
            except Exception:
                pass
        if idx_val is None:
            m = re.search(r"-?\d+", out)
            if m:
                try:
                    idx_val = int(m.group(0))
                except Exception:
                    idx_val = None
        if idx_val is None:
            if _VERBOSE:
                print("pick parse failed: ", out[:120].replace("\n"," "))
            continue
        if idx_val == len(contents):
            idx_val -= 1
        # Map to nearest allowed candidate if not directly allowed
        if idx_val not in candidates:
            guess = idx_val - 1 if (idx_val - 1) in candidates else idx_val
            if candidates:
                idx_val = min(candidates, key=lambda c: (abs(c - guess), c))
        if 0 <= idx_val < len(contents):
            chosen[ci] = idx_val
    return chosen


_gpt_corrupt_llm: Optional[CachingLLM] = None


def get_gpt_corrupt_llm(model: str = "gpt-4o", cache_file: str = "openai_ai_llm_cache_sampler.pkl") -> CachingLLM:
    global _gpt_corrupt_llm
    if _gpt_corrupt_llm is None or _gpt_corrupt_llm.model_params.get("model") != model or getattr(_gpt_corrupt_llm, 'cache_file', None) != cache_file:
        system_prompt = (
            "You are given one proof line. Change the line so that it becomes incorrect but still mathematically sensible."
            "Do NOT change numbering, labels, formatting-only tokens, or spacing. "
            "Try to keep the line very similar to the original line (so if there are markdown tokens for closing a line or spacing, keep those)."
            "Return ONLY the modified single line (no commentary, no extra lines)."
            "Make only ONE single change for the line. Change only ONE mathematical property or relation, do not do multiple changes."
        )
        _gpt_corrupt_llm = CachingLLM(
            model_params={"model": "o3-mini", "reasoning_effort": "high"},
            initial_messages=[{"role": "system", "content": system_prompt}],
            cache_file=cache_file,
        )
    return _gpt_corrupt_llm


def batch_gpt_corrupt_lines(lines: List[str], model: str = "gpt-4o", cache_file: str = "openai_ai_llm_cache_sampler.pkl") -> List[Optional[str]]:
    if not lines:
        return []
    prompts = [f"{ln}" for ln in lines]
    try:
        resps = get_gpt_corrupt_llm(model, cache_file=cache_file).generate_responses(prompts)
    except Exception:
        return [None] * len(lines)
    out: List[Optional[str]] = []
    for ln, r in zip(lines, resps):
        text = get_content_from_response(r.to_dict())
        if text is None:
            out.append(None)
            continue
        text = text.replace("\n", " ").strip()
        if text and text != ln:
            out.append(text)
        else:
            out.append(None)
    return out


def pick_lines_batched(
    contents_list: List[List[str]],
    model: str,
    cache_file: str,
    batch_size: int,
) -> List[Optional[int]]:
    """Call batch_gpt_pick_lines in small chunks to avoid cache collisions."""
    chosen: List[Optional[int]] = []
    step = max(1, int(batch_size))
    for i in range(0, len(contents_list), step):
        chunk = contents_list[i : i + step]
        chunk_choices = batch_gpt_pick_lines(chunk, model=model, cache_file=cache_file)
        chosen.extend(chunk_choices)
    return chosen


def corrupt_lines_batched(
    lines: List[str],
    model: str,
    cache_file: str,
    batch_size: int,
) -> List[Optional[str]]:
    """Call batch_gpt_corrupt_lines in small chunks to avoid cache collisions."""
    out: List[Optional[str]] = []
    step = max(1, int(batch_size))
    for i in range(0, len(lines), step):
        chunk = lines[i : i + step]
        out.extend(batch_gpt_corrupt_lines(chunk, model=model, cache_file=cache_file))
    return out


def try_corrupt_with_fallback(
    contents: List[str],
    initial_idx: Optional[int],
    cache_file: str,
    model: str = "gpt-4o",
    max_tries: int = 6,
) -> Optional[Tuple[int, str]]:
    """Attempt corruption starting from initial_idx; if it fails, try other substantial indices.
    Returns (idx, modified_line) or None if all attempts fail.
    """
    candidates = line_indices_substantial(contents)
    if not candidates:
        return None
    order: List[int] = []
    if initial_idx is not None and initial_idx in candidates:
        order.append(initial_idx)
    for i in candidates:
        if i not in order:
            order.append(i)
    # Limit tries
    order = order[:max_tries]
    for idx in order:
        orig = contents[idx]
        mods = batch_gpt_corrupt_lines([orig], model=model, cache_file=cache_file)
        mod = mods[0] if mods else None
        if is_meaningful_change(orig, mod):
            return idx, mod
    return None


_gpt_pick_llm: Optional[CachingLLM] = None


def get_gpt_pick_llm(model: str = "gpt-4o", cache_file: str = "openai_ai_llm_cache_sampler.pkl") -> CachingLLM:
    global _gpt_pick_llm
    if _gpt_pick_llm is None or _gpt_pick_llm.model_params.get("model") != model or getattr(_gpt_pick_llm, 'cache_file', None) != cache_file:
        system_prompt = "Return only the requested JSON."
        _gpt_pick_llm = CachingLLM(
            model_params={"model": model, "temperature": 0.0},
            initial_messages=[{"role": "system", "content": system_prompt}],
            cache_file=cache_file,
        )
    return _gpt_pick_llm




def compute_adaptive_bins(
    proofs: List[ProofRef],
    num_bins: int,
    min_samples_per_bin: int,
) -> List[Tuple[float, float]]:
    """Compute bin boundaries that ensure each bin has at least min_samples_per_bin proofs.

    Strategy:
    1. Start with log-spaced bin edges
    2. Count proofs in each bin
    3. Merge adjacent bins that have too few proofs
    4. Expand bins at the ends if still insufficient

    Returns list of (lower_bound, upper_bound) tuples for each final bin.
    """
    lengths = sorted([len(p.contents) for p in proofs])
    min_len = lengths[0]
    max_len = lengths[-1]

    # Start with log-spaced edges
    initial_edges = [
        math.exp(math.log(min_len) + i * (math.log(max_len) - math.log(min_len)) / float(num_bins))
        for i in range(num_bins)
    ] + [float(max_len)]

    # Count proofs in each initial bin
    def count_in_range(lo, hi, is_last=False):
        if is_last:
            return sum(1 for L in lengths if lo <= L <= hi)
        else:
            return sum(1 for L in lengths if lo <= L < hi)

    # Build bins, merging those with insufficient proofs
    bins = []
    i = 0
    while i < num_bins:
        lo = initial_edges[i]
        hi = initial_edges[i + 1]
        is_last = (i == num_bins - 1)
        count = count_in_range(lo, hi, is_last)

        # Track starting position
        start_i = i

        # Merge forward while count is insufficient
        while count < min_samples_per_bin and i + 1 < num_bins:
            i += 1
            hi = initial_edges[i + 1]
            is_last = (i == num_bins - 1)
            count = count_in_range(lo, hi, is_last)

        bins.append((lo, hi))
        # Move to next bin after the last one we merged
        # If we merged bins start_i through i, next bin is i+1
        i += 1

    # Second pass: merge any remaining bins that are still too small
    # Merge backwards since forward merging is already done
    merged_bins = []
    i = 0
    while i < len(bins):
        lo, hi = bins[i]
        is_last = (i == len(bins) - 1)
        count = count_in_range(lo, hi, is_last)

        # If this bin is too small and not the first, merge with previous
        if count < min_samples_per_bin and len(merged_bins) > 0:
            # Merge with previous bin
            prev_lo, prev_hi = merged_bins[-1]
            merged_bins[-1] = (prev_lo, hi)
        else:
            merged_bins.append((lo, hi))

        i += 1

    bins = merged_bins

    # Final validation - if any bin still has <min_samples_per_bin, error out
    final_counts = []
    for j, (lo, hi) in enumerate(bins):
        is_last = (j == len(bins) - 1)
        count = count_in_range(lo, hi, is_last)
        final_counts.append(count)

    insufficient = [(j, count) for j, count in enumerate(final_counts) if count < min_samples_per_bin]

    if insufficient:
        msg = f"ERROR: After adaptive binning, {len(insufficient)} bins still have <{min_samples_per_bin} proofs:\n"
        for bin_idx, count in insufficient[:5]:
            lo, hi = bins[bin_idx]
            msg += f"  Bin {bin_idx}: [{int(lo)}, {int(hi)}] has {count} proofs\n"
        msg += f"\nTotal proofs: {len(proofs)}\n"
        msg += f"Requested {num_bins} bins with {min_samples_per_bin} min each = {num_bins * min_samples_per_bin} proofs needed\n"
        msg += f"\nSuggestions:\n"
        msg += f"  1. Use fewer bins: --bins {max(1, num_bins // 2)}\n"
        msg += f"  2. Reduce min samples: --min-samples-per-bin {min(final_counts)}\n"
        raise ValueError(msg)

    return bins


def sample_stratified(
    proofs: List[ProofRef],
    n_samples: int,
    seed: int,
    n_bins: Optional[int] = None,
    hard_bins: bool = False,
    min_samples_per_bin: int = 10,
) -> Tuple[List[Tuple[ProofRef, int]], Optional[List[float]]]:
    """Stratified sampling that captures min and max proof lengths, with log-scale targets in between.

    - Always tries to include one sample at the minimum length and one at the maximum length (when n_samples >= 2).
    - For the remaining samples, uses log-spaced targets between min_len and max_len and picks the closest available lengths.
    - If duplicates/exhaustion occur, fills from remaining proofs at random.
    - Returns a list of (ProofRef, bin_idx) where bin_idx indicates target order: 0=min, last=max, and intermediates are 1..k.
    - When hard_bins=True and min_samples_per_bin is set, bins are adaptively sized to ensure minimum coverage.
    """
    rng = random.Random(seed)
    if not proofs or n_samples <= 0:
        return [], None

    lengths = [len(p.contents) for p in proofs]
    # Map length -> list of indices with that length
    length_to_indices: Dict[int, List[int]] = {}
    for i, l in enumerate(lengths):
        length_to_indices.setdefault(l, []).append(i)

    unique_lengths = sorted(length_to_indices.keys())
    min_len = unique_lengths[0]
    max_len = unique_lengths[-1]

    if hard_bins:
        # Use adaptive bin sizing to ensure min_samples_per_bin
        if n_bins is None or n_bins <= 0:
            num_bins = min(n_samples, max(1, len(unique_lengths)))
        else:
            num_bins = int(n_bins)

        if num_bins == 1:
            target = math.exp((math.log(max(1, min_len)) + math.log(max_len)) / 2.0)
            nearest = min(unique_lengths, key=lambda L: abs(L - target))
            idx = length_to_indices[nearest].pop(rng.randrange(len(length_to_indices[nearest])))
            return [(proofs[idx], 0)], [float(min_len), float(max_len)]

        # Compute adaptive bins that ensure each has at least min_samples_per_bin proofs
        requested_bins = num_bins
        adaptive_bins = compute_adaptive_bins(proofs, num_bins, min_samples_per_bin)
        num_bins = len(adaptive_bins)  # May be fewer after merging

        if num_bins < requested_bins:
            print(f"INFO: Adaptive binning merged {requested_bins} requested bins into {num_bins} final bins")
            print(f"      to ensure each bin has ≥{min_samples_per_bin} proofs.")
            print(f"      Distributing {n_samples} samples uniformly: ~{n_samples // num_bins} per bin")

        # Assign proofs to adaptive bins
        bins_lists: List[List[int]] = [[] for _ in range(num_bins)]
        for idx, L in enumerate(lengths):
            for i, (lo, hi) in enumerate(adaptive_bins):
                is_last = (i == num_bins - 1)
                if is_last:
                    if L >= lo and L <= hi:
                        bins_lists[i].append(idx)
                        break
                else:
                    if L >= lo and L < hi:
                        bins_lists[i].append(idx)
                        break

        # Convert adaptive_bins to edges format for return value
        edges = [lo for lo, _ in adaptive_bins] + [adaptive_bins[-1][1]]

        # Uniform sampling: allocate samples evenly across bins
        # After merging, num_bins may be less than requested, so distribute n_samples uniformly
        samples_per_bin = n_samples // num_bins
        extra_samples = n_samples % num_bins

        chosen: List[Tuple[ProofRef, int]] = []
        used = set()

        for bin_idx in range(num_bins):
            # Determine how many samples this bin should get
            target_for_bin = samples_per_bin + (1 if bin_idx < extra_samples else 0)

            # Sample from this bin
            candidates = [i for i in bins_lists[bin_idx] if i not in used]
            if len(candidates) < target_for_bin:
                # Not enough proofs in this bin - take all available
                selected = candidates
            else:
                # Randomly sample target_for_bin proofs from this bin
                selected = rng.sample(candidates, target_for_bin)

            for i in selected:
                used.add(i)
                chosen.append((proofs[i], bin_idx))

        # If we're still short (shouldn't happen with proper min_samples_per_bin), use old logic
        if len(chosen) < n_samples:
            # Fallback: round-robin to fill remaining
            remaining = n_samples - len(chosen)
            b = 0
            while remaining > 0:
                candidates = [i for i in bins_lists[b] if i not in used]
                if candidates:
                    i = rng.choice(candidates)
                    used.add(i)
                    chosen.append((proofs[i], b))
                    remaining -= 1
                b = (b + 1) % num_bins
                if b == 0 and all(not [i for i in bl if i not in used] for bl in bins_lists):
                    break

        # If still short, fill from nearest-to-center per bin first, then globally
        if len(chosen) < n_samples:
            for bi in range(num_bins):
                if len(chosen) >= n_samples:
                    break
                rem = [i for i in bins_lists[bi] if i not in used]
                if not rem:
                    continue
                lo = edges[bi]
                hi = edges[bi + 1]
                center = (lo + hi) / 2.0
                i = min(rem, key=lambda j: abs(lengths[j] - center))
                used.add(i)
                chosen.append((proofs[i], bi))
                if len(chosen) >= n_samples:
                    break
        if len(chosen) < n_samples:
            remaining = [i for i in range(len(proofs)) if i not in used]
            rng.shuffle(remaining)
            for i in remaining:
                if len(chosen) >= n_samples:
                    break
                # assign to nearest bin by center for labeling
                centers = [ ( (edges[k]+edges[k+1])/2.0, k) for k in range(num_bins) ]
                bi = min(centers, key=lambda t: abs(lengths[i] - t[0]))[1]
                used.add(i)
                chosen.append((proofs[i], bi))

        return chosen[:n_samples], edges

    def pop_index_for_length(L: int) -> Optional[int]:
        arr = length_to_indices.get(L)
        if arr:
            return arr.pop(rng.randrange(len(arr)))
        return None

    def nearest_available_length(target: float) -> Optional[int]:
        # Find nearest length in unique_lengths that still has available indices
        # Binary search then expand outwards
        lo, hi = 0, len(unique_lengths) - 1
        # Position by value
        while lo <= hi:
            mid = (lo + hi) // 2
            if unique_lengths[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        # Candidates around insertion point lo
        left = lo - 1
        right = lo
        best_len = None
        best_dist = None
        # Search window up to len(unique_lengths)
        steps = 0
        while (left >= 0 or right < len(unique_lengths)) and steps < len(unique_lengths):
            steps += 1
            # Check left
            if left >= 0:
                L = unique_lengths[left]
                if length_to_indices.get(L):
                    d = abs(L - target)
                    if best_dist is None or d < best_dist:
                        best_len = L
                        best_dist = d
            # Check right
            if right < len(unique_lengths):
                L = unique_lengths[right]
                if length_to_indices.get(L):
                    d = abs(L - target)
                    if best_dist is None or d < best_dist:
                        best_len = L
                        best_dist = d
            left -= 1
            right += 1
        return best_len

    chosen: List[Tuple[ProofRef, int]] = []
    used_idx = set()

    def add_by_length(L: int, bin_idx: int):
        idx = pop_index_for_length(L)
        if idx is not None and idx not in used_idx:
            used_idx.add(idx)
            chosen.append((proofs[idx], bin_idx))

    # Edge cases: if only one sample requested
    if n_samples == 1:
        # Choose length closest to geometric mean
        target = math.exp((math.log(max(1, min_len)) + math.log(max_len)) / 2.0)
        L = nearest_available_length(target) or min_len
        add_by_length(L, 0)
        return chosen, None

    # Include min and max first (if possible)
    add_by_length(min_len, 0)
    if len(chosen) < n_samples:
        add_by_length(max_len, n_samples - 1)

    remaining_needed = n_samples - len(chosen)
    if remaining_needed > 0 and max_len > 0 and min_len > 0 and max_len != min_len:
        # Log-spaced targets strictly between min and max
        log_min = math.log(min_len)
        log_max = math.log(max_len)
        # Create remaining_needed targets between (exclusive) ends
        for k in range(1, remaining_needed + 1):
            frac = k / (remaining_needed + 1)
            target = math.exp(log_min + frac * (log_max - log_min))
            L = nearest_available_length(target)
            if L is None:
                continue
            # bin_idx place them after min
            add_by_length(L, k)

    # If still short, fill randomly from whatever remains
    if len(chosen) < n_samples:
        remaining = [i for i in range(len(proofs)) if i not in used_idx]
        rng.shuffle(remaining)
        while remaining and len(chosen) < n_samples:
            idx = remaining.pop()
            used_idx.add(idx)
            # bin_idx -1 indicates random fill
            chosen.append((proofs[idx], -1))

    # If we ended up with more than requested due to edge cases, trim
    return chosen[:n_samples], None


def run(
    seed: int,
    n: int,
    out_path: str,
    bins: Optional[int],
    use_gpt: bool,
    min_lines: int = 1,
    max_topup_batches: int = 3,
    fail_fast: bool = False,
    llm_cache_file: str = "openai_ai_llm_cache_sampler_2.pkl",
    pick_batch_size: int = 1,
    corrupt_batch_size: int = 100,
    hard_bins: bool = False,
    min_samples_per_bin: int = 10,
) -> Dict[str, Any]:
    proofs = load_datasets()
    if not proofs:
        raise SystemExit("No proofs loaded. Ensure NaturalProofs JSON files exist in current directory.")

    # Filter by minimum proof length
    proofs = [p for p in proofs if len(p.contents) >= min_lines]
    if not proofs:
        raise SystemExit(f"No proofs meet the minimum line requirement (min_lines={min_lines}).")

    # Sample proofs
    picked, bin_edges = sample_stratified(proofs, n, seed, n_bins=bins, hard_bins=hard_bins, min_samples_per_bin=min_samples_per_bin)
    bin_boundaries = None
    if hard_bins and bin_edges:
        # Convert float edges to non-overlapping, contiguous integer bins [lo, hi]
        import math
        B = len(bin_edges) - 1
        if B > 0:
            min_len_i = int(math.floor(bin_edges[0]))
            max_len_i = int(math.ceil(bin_edges[-1]))
            lo = min_len_i
            bin_boundaries = []
            for i in range(B):
                if i < B - 1:
                    # upper bound exclusive edge becomes inclusive integer hi using floor
                    hi = int(math.floor(bin_edges[i + 1]))
                else:
                    hi = max_len_i
                # clamp and enforce contiguity/non-overlap
                if hi < lo:
                    hi = lo
                if hi > max_len_i:
                    hi = max_len_i
                bin_boundaries.append([lo, hi])
                lo = hi + 1
            # Print integer bin boundaries for visibility
            print(f"hard-bins integer boundaries (inclusive): {bin_boundaries}")
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    chosen_keys = set()
    if use_gpt:
        # Two-stage: pick lines then corrupt them
        contents_list = [ref.contents for ref, _ in picked]
        chosen_idxs = pick_lines_batched(contents_list, model="gpt-4o", cache_file=llm_cache_file, batch_size=pick_batch_size)
        if _VERBOSE:
            print(f"gpt-pick: requested={len(contents_list)} chosen={sum(1 for i in chosen_idxs if i is not None)}")
        to_corrupt: List[str] = []
        back_refs: List[Tuple[ProofRef, int, int, List[str]]] = []  # (ref, bin, idx, proof_lines)
        for (ref, bin_idx), idx in zip(picked, chosen_idxs):
            if idx is None:
                continue
            to_corrupt.append(ref.contents[idx])
            back_refs.append((ref, bin_idx, idx, ref.contents))
        if not to_corrupt and fail_fast:
            if _VERBOSE:
                print("No eligible GPT picks in initial batch; failing fast.")
            return {"seed": seed, "n": n, "use_gpt": use_gpt, "min_lines": min_lines, "samples": []}
        mods = corrupt_lines_batched(to_corrupt, model="o3-mini", cache_file=llm_cache_file, batch_size=corrupt_batch_size)
        if _VERBOSE:
            print(f"gpt-corrupt: requested={len(to_corrupt)} modified={sum(1 for m in mods if m is not None)}")

        # First pass: identify which corruptions need fallback
        fallback_needed = []
        valid_corruptions = []
        fail_reasons = {"none": 0, "unchanged": 0, "semantic_only": 0}

        for i, ((ref, bin_idx, idx, proof_lines), mod) in enumerate(zip(back_refs, mods)):
            if not is_meaningful_change(proof_lines[idx], mod):
                # Determine failure reason for logging
                if not mod:
                    fail_reasons["none"] += 1
                elif mod == proof_lines[idx]:
                    fail_reasons["unchanged"] += 1
                else:
                    fail_reasons["semantic_only"] += 1
                fallback_needed.append((i, ref, bin_idx, idx, proof_lines))
            else:
                valid_corruptions.append((i, ref, bin_idx, idx, proof_lines, mod))

        if _VERBOSE and fallback_needed:
            print(f"Validation failures: None={fail_reasons['none']}, Unchanged={fail_reasons['unchanged']}, SemanticOnly={fail_reasons['semantic_only']}")

        # Batched fallback processing for failed corruptions
        fallback_results = {}  # Maps original index -> (new_idx, new_mod)
        if fallback_needed:
            if _VERBOSE:
                print(f"Batched fallback: {len(fallback_needed)} samples need fallback corruption")
            fallback_lines = []
            fallback_map = []  # Maps fallback_lines index -> (original_i, candidate_idx)

            for i, ref, bin_idx, idx, proof_lines in fallback_needed:
                candidates = line_indices_substantial(proof_lines)
                if not candidates:
                    continue
                # Try up to 6 candidates per proof
                for candidate_idx in candidates[:6]:
                    fallback_lines.append(proof_lines[candidate_idx])
                    fallback_map.append((i, candidate_idx))

            if fallback_lines:
                fallback_mods = corrupt_lines_batched(fallback_lines, model="o3-mini", cache_file=llm_cache_file, batch_size=corrupt_batch_size)

                # Process fallback results - take first valid corruption for each original sample
                for (orig_i, cand_idx), fb_mod in zip(fallback_map, fallback_mods):
                    if orig_i in fallback_results:
                        continue  # Already found a valid corruption for this sample

                    orig_line = fallback_lines[fallback_map.index((orig_i, cand_idx))]
                    if is_meaningful_change(orig_line, fb_mod):
                        fallback_results[orig_i] = (cand_idx, fb_mod)

        # Combine valid corruptions and successful fallbacks
        all_corruptions = []
        for i, ref, bin_idx, idx, proof_lines, mod in valid_corruptions:
            all_corruptions.append((ref, bin_idx, idx, proof_lines, mod))

        for i, ref, bin_idx, idx, proof_lines in fallback_needed:
            if i in fallback_results:
                new_idx, new_mod = fallback_results[i]
                all_corruptions.append((ref, bin_idx, new_idx, proof_lines, new_mod))

        # Build final results
        if _VERBOSE:
            print(f"Final: {len(all_corruptions)}/{len(back_refs)} corruptions successful")

        for ref, bin_idx, idx, proof_lines, mod in all_corruptions:
            modified_proof = list(proof_lines)
            modified_proof[idx] = mod
            results.append(
                {
                    "source": ref.source,
                    "theorem_id": ref.theorem_id,
                    "theorem_title": ref.theorem_title,
                    "proof_index": ref.proof_index,
                    "proof_length": len(ref.contents),
                    "bin_index": bin_idx,
                    "chosen_line_index": idx,
                    "original_line": proof_lines[idx],
                    "modified_line": mod,
                    "original_proof": ref.contents,
                    "modified_proof": modified_proof,
                    "selection_method": "gpt-pick",
                    "corruption_method": "gpt-corrupt",
                }
            )
            chosen_keys.add((ref.source, ref.theorem_id, ref.proof_index))

    # If we didn't reach n due to filters, top up with random unique proofs
    if len(results) < n:
        rng2 = random.Random(seed + 1)
        indices = list(range(len(proofs)))
        rng2.shuffle(indices)
        # Prepare batch GPT for the remaining set (two-stage in chunks)
        remaining_refs = [
            proofs[i]
            for i in indices
            if (proofs[i].source, proofs[i].theorem_id, proofs[i].proof_index) not in chosen_keys
        ]
        if use_gpt:
            cursor = 0
            batch_count = 0
            while len(results) < n and cursor < len(remaining_refs):
                if batch_count >= max_topup_batches:
                    if _VERBOSE:
                        print(f"Stopping top-up after {batch_count} batches (max_topup_batches reached)")
                    break
                needed = n - len(results)
                # Honor user-configured batch sizes for pick/corrupt; cap by remaining size
                target_batch = max(pick_batch_size, corrupt_batch_size)
                chunk_size = min(max(needed, 1), target_batch)
                chunk = remaining_refs[cursor : cursor + chunk_size]
                cursor += chunk_size
                if not chunk:
                    break
                chunk_contents = [r.contents for r in chunk]
                chosen_idxs = pick_lines_batched(chunk_contents, model="gpt-4o", cache_file=llm_cache_file, batch_size=pick_batch_size)
                to_corrupt: List[str] = []
                back_refs: List[Tuple[ProofRef, int, List[str]]] = []  # (ref, idx, proof_lines)
                for ref, idx in zip(chunk, chosen_idxs):
                    if idx is None:
                        continue
                    to_corrupt.append(ref.contents[idx])
                    back_refs.append((ref, idx, ref.contents))
                if not to_corrupt:
                    if _VERBOSE:
                        print("Top-up batch produced no picks; moving to next chunk")
                    batch_count += 1
                    continue
                mods = corrupt_lines_batched(to_corrupt, model="gpt-4o", cache_file=llm_cache_file, batch_size=corrupt_batch_size)
                if _VERBOSE:
                    print(f"topup batch: pick_chosen={sum(1 for i in chosen_idxs if i is not None)} corrupt_ok={sum(1 for m in mods if m is not None)} results={len(results)}")
                for (ref, idx, proof_lines), mod in zip(back_refs, mods):
                    if len(results) >= n:
                        break
                    if mod is None:
                        continue
                    key = (ref.source, ref.theorem_id, ref.proof_index)
                    if key in chosen_keys:
                        continue
                    orig = proof_lines[idx]
                    if mod == orig or (
                        extract_semantic_tokens(mod) == extract_semantic_tokens(orig) and not numbers_changed(orig, mod)
                    ):
                        fb = try_corrupt_with_fallback(proof_lines, initial_idx=None, cache_file=llm_cache_file, model="gpt-4o")
                        if not fb:
                            continue
                        idx, mod = fb
                        orig = proof_lines[idx]
                    modified_proof = list(proof_lines)
                    modified_proof[idx] = mod
                    results.append(
                        {
                            "source": ref.source,
                            "theorem_id": ref.theorem_id,
                            "theorem_title": ref.theorem_title,
                            "proof_index": ref.proof_index,
                            "proof_length": len(ref.contents),
                            "bin_index": None,
                            "chosen_line_index": idx,
                            "original_line": orig,
                            "modified_line": mod,
                            "original_proof": ref.contents,
                            "modified_proof": modified_proof,
                            "selection_method": "gpt-pick",
                            "corruption_method": "gpt-corrupt",
                        }
                    )
                    chosen_keys.add(key)

    output = {"seed": seed, "n": n, "use_gpt": use_gpt, "min_lines": min_lines, "samples": results}
    if bin_boundaries is not None:
        output["bin_boundaries_inclusive"] = bin_boundaries

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def main():
    parser = argparse.ArgumentParser(description="Sample and corrupt proof lines from NaturalProofs datasets.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--n", type=int, default=200, help="Number of samples to select")
    parser.add_argument(
        "--bins", type=int, default=20, help="Number of stratification bins (defaults to n)"
    )
    parser.add_argument(
        "--hard-bins",
        action="store_true",
        default=True,
        help="Use log-spaced hard bin edges to sample one proof per bin (includes extremes)",
    )
    parser.add_argument(
        "--min-samples-per-bin",
        type=int,
        default=10,
        help="Minimum number of proofs required in each bin (adaptive binning will merge bins if needed)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="modified_samples_min_rangev1_X3_gpt.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--use-gpt",
        action="store_true",
        default=True,
        help="Use OpenAI gpt-4o to pick consequential lines and corrupt them (two-stage, no heuristic)",
    )
    parser.add_argument(
        "--max-topup-batches",
        type=int,
        default=3,
        help="Maximum number of GPT top-up batches to try when results < n",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit early with 0 samples if the first GPT batch yields no picks",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=5,
        help="Minimum number of lines a proof must have to be eligible for sampling",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="openai_ai_llm_cache_sampler_3.pkl",
        help="Cache file to use for GPT calls (avoid colliding with other runs)",
    )
    parser.add_argument(
        "--pick-batch-size",
        type=int,
        default=100,
        help="Batch size for GPT pick stage (default 1 to avoid cache issues)",
    )
    parser.add_argument(
        "--corrupt-batch-size",
        type=int,
        default=100,
        help="Batch size for GPT corrupt stage (default 1 to avoid cache issues)",
    )

    args = parser.parse_args()

    run(
        seed=args.seed,
        n=args.n,
        out_path=args.out,
        bins=args.bins,
        use_gpt=args.use_gpt,
        min_lines=args.min_lines,
        max_topup_batches=args.max_topup_batches,
        fail_fast=args.fail_fast,
        llm_cache_file=args.cache_file,
        pick_batch_size=args.pick_batch_size,
        corrupt_batch_size=args.corrupt_batch_size,
        hard_bins=args.hard_bins,
        min_samples_per_bin=args.min_samples_per_bin,
    )


if __name__ == "__main__":
    main()
