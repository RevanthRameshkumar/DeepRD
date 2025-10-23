# DeepRD
Evaluation toolset for LLMs using parameterized graph complexity

# Citation
This repo contains code to perform the experiments and analysis in our paper: "Reasoning Models Reason Well, Until They Don’t"



If you use this data, analysis, or code in your work, please cite:

[bibtex citation coming soon!]
Reasoning Models Reason Well, Until They Don’t
Revanth Rameshkumar, Jimson Huang, Yunxin Sun, Fei Xia, Abulhair Saparov
{revr,fxia}@uw.edu {huan2073,sun1114,asaparov}@purdue.edu


# Graph Dataset Generator (in /graph_generator)
## Usage

```
python3 graph_generator.py --help

Required Arguments:
  -l, --lookaheads L1 L2 ...    Lookahead values (minimum path lengths)
  -b, --branches B1 B2 ...      Branch values (outgoing edges from start node)

Optional Arguments:
  -n, --samples N               Samples per combination (default: 10)
  -o, --output FILE             Output filename (default: auto-generated)
  --seed SEED                   Random seed (default: 9, same as paper)
  --mode {symbolic,logic}       Output format: symbolic or logic (default: symbolic)
  --no-dedupe                   Disable graph deduplication
  --no-enforce-branches         Disable strict branching enforcement
  -q, --quiet                   Suppress progress output
```

## Quick Start
```bash
# Generate a small symbolic (graph path) dataset
python3 graph_generator.py -l 2 4 -b 1 2 --samples 5 -o test_symbolic.json

# Generate a small logic (proof next step) dataset
python3 graph_generator.py -l 2 4 -b 1 2 --samples 5 --mode logic -o test_logic.json

# Generate paper's full symbolic dataset (branching)
python3 graph_generator.py \
  -l 2 3 4 5 6 7 8 9 10 16 32 64 100 128 150 200 250 256 300 350 512 800 \
  -b 2 4 8 16 \
  --samples 10 \
  -o branching_dataset.json

# Generate paper's full symbolic dataset (trivial/linear)
python3 graph_generator.py \
  -l 2 3 4 5 6 7 8 9 10 16 32 64 100 128 150 200 250 256 300 350 512 1024 1536 \
  -b 1 \
  --samples 10 \
  -o trivial_dataset.json

# Generate logic dataset from same graph parameters
python3 graph_generator.py \
  -l 2 3 4 5 6 7 8 9 10 16 32 64 100 128 150 200 250 256 300 350 512 800 \
  -b 2 4 8 16 \
  --samples 10 \
  --mode logic \
  -o logic_dataset.json
```

## Output Format

**Symbolic mode (default)** - Graph path finding:
```json
{
  "edges": [[1,3], [3,7], ...],      // Directed edge list
  "query": [1, 8],                    // Start and end nodes
  "lookahead_size": 2,                // Lookahead values
  "max_branches": 2,                  // Branching factor
  "question": "Determine if there..." // Full question text
}
```

**Logic mode** - Proofs next step:
```json
{
  "edges": [[1,2], [2,3]],
  "query": [1, 3],
  "lookahead_size": 2,
  "max_branches": 1,
  "question": "Determine if there...",
  "logic_predicates": "Given the following list of predicates:\nIf someone is X, they are Y...",
  "logical_question": "Given that James is X, and we want to prove James is Z...",
  "next_adjective": ["Y"],           // Correct answer(s)
  "node_mapping": {                  // Node ID -> name and predicate mapping
    "1": {"name": "James", "adjective": "X"},
    "2": {"name": "James", "adjective": "Y"}
  }
}
```


# LLM Prediction (in /llm_prediction)

Run LLM inference on graph datasets to reproduce the paper's results.

## Setup

1. **Configure API Keys**: Copy `.env.template` to `.env` and add your API keys:
   ```bash
   cp .env.template .env
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=your_key_here
   # TOGETHER_API_KEY=your_key_here
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
cd llm_prediction

# Symbolic reasoning (graph path finding)
python llm_prediction.py \
  --mode symbolic \
  --model o3-mini \
  --input-file ../graph_generator/branching_dataset.json \
  --reasoning-effort medium \
  --output-dir ../llm_results/symbolic

# Logic reasoning (proof next step prediction)
python llm_prediction.py \
  --mode logic \
  --model r1 \
  --input-file ../graph_generator/logic_dataset.json \
  --output-dir ../llm_results/logic
```

**Arguments:**
- `--mode {symbolic,logic}`: Reasoning task type
  - `symbolic`: Graph path finding (uses |YES|/|NO| prompts)
  - `logic`: Logical next step prediction (uses fill-in-blank prompts)
- `--model {o3,o3-mini,r1,v3,gpt4o,gpt4}`: LLM model to use
  - `o3`, `o3-mini`: OpenAI o3 models (requires `--reasoning-effort`)
  - `r1`, `v3`: DeepSeek R1/V3 (Together AI)
  - `gpt4o`, `gpt4`: GPT-4 variants (OpenAI)
- `--input-file FILE`: Input JSON from graph_generator.py
- `--output-dir DIR`: Directory for results (default: current)
- `--cache-dir DIR`: Cache directory (default: current)
- `--reasoning-effort {low,medium,high}`: For o3 models (default: medium)

**Output:**
- `FINAL_llm_results_{timestamp}_{model}.json` (symbolic mode)
- `FINAL_LOGIC_llm_results_{timestamp}_{model}.json` (logic mode)

Each output contains LLM responses and extracted paths/answers.


# Experimental Results and Metrics

This repository contains the final experimental results from the paper in the `llm_results/` directory.

## Symbolic Reasoning Results (Graph Path Finding) in `llm_results/symbolic`

| Model | File |
|-------|------|
| DeepSeek V3 | `FINAL_llm_results_20250724_230912_deepseek-ai_DeepSeek-V3.json` |
| DeepSeek R1 | `FINAL_llm_results_20250724_222511_deepseek-ai_DeepSeek-R1.json` |
| GPT-4o | `FINAL_llm_results_20250724_220322_gpt-4o.json` |
| o3-mini | `FINAL_llm_results_20250724_215314_o3-mini.json` |
| o3 | `FINAL_llm_results_20250724_234305_o3.json` |

## Logic Reasoning Results in `llm_results/logic`

| Model | File |
|-------|------|
| DeepSeek V3 | `FINAL_LOGIC_llm_results_20250725_004739_deepseek-ai_DeepSeek-V3.json` |
| DeepSeek R1 | `FINAL_LOGIC_llm_results_20250725_112013_deepseek-ai_DeepSeek-R1.json` |
| GPT-4o | `FINAL_LOGIC_llm_results_20250725_004033_gpt-4o.json` |
| o3-mini | `FINAL_LOGIC_llm_results_20250725_002037_o3-mini.json` |
| o3 | `FINAL_LOGIC_llm_results_20250725_005647_o3.json` |

## Data Format

Each JSON file contains a list of samples with the following structure:

```json
{
  "edges": [[1, 3], [3, 7], ...],           // Graph edges
  "query": [1, 8],                          // Start and end nodes
  "lookahead_size": 2,                      // Minimum path length
  "max_branches": 2,                        // Branching factor
  "question": "Determine if there is...",   // Full question prompt
  "llm_response": "...",                    // Model's response
  "extracted_path": [[1, 3, 7, 8]]          // Extracted path from response
}
```

## Analyzing Results

Use `llm_results/llm_metrics.py` to generate plots and metrics tables from the result files:

```bash
# Analyze symbolic reasoning results
python3 llm_results/llm_metrics.py \
  --files llm_results/symbolic/FINAL_llm_results_20250724_222511_deepseek-ai_DeepSeek-R1.json \
          llm_results/symbolic/FINAL_llm_results_20250724_215314_o3-mini.json \
  --models R1 o3-mini \
  --mode symbolic \
  --output-dir output/symbolic

# Analyze logic reasoning results
python3 llm_results/llm_metrics.py \
  --files llm_results/logic/FINAL_LOGIC_llm_results_20250725_112013_deepseek-ai_DeepSeek-R1.json \
          llm_results/logic/FINAL_LOGIC_llm_results_20250725_002037_o3-mini.json \
  --models R1 o3-mini \
  --mode logic \
  --output-dir output/logic

# Compare all models on symbolic reasoning
python3 llm_results/llm_metrics.py \
  --files llm_results/symbolic/FINAL_llm_results_*_DeepSeek-V3.json \
          llm_results/symbolic/FINAL_llm_results_*_DeepSeek-R1.json \
          llm_results/symbolic/FINAL_llm_results_*_gpt-4o.json \
          llm_results/symbolic/FINAL_llm_results_*_o3-mini.json \
  --models V3 R1 4o o3-mini \
  --mode symbolic
```

**Output:**
- TSV files: `{mode}_{model}_metrics.tsv` with aggregated metrics per (lookahead, branches) combination
- PDF plots:
  - **Accuracy plots:**
    - `{mode}_full_path_accuracy_branches_2_4_8_16.pdf` - 4-panel full path accuracy
    - `{mode}_first_node_accuracy_branches_2_4_8_16.pdf` - 4-panel first step accuracy
    - `{mode}_first_node_accuracy_branches_1.pdf` - First step accuracy for branch=1
    - `{mode}_valid_path_ratio_branch_1.pdf` - Full path accuracy for branch=1
  - **Token usage plots:**
    - `{mode}_avg_tokens_branches_2_4_8_16.pdf` - 4-panel average token usage
    - `{mode}_avg_tokens_branches_8.pdf` - Token usage for branch=8
    - `{mode}_avg_tokens_branches_1.pdf` - Token usage for branch=1
  - **Error analysis plots:**
    - `{mode}_edge_hallucinations_branches_2_4_8_16.pdf` - 4-panel hallucination rates
    - `{mode}_edge_hallucinations_branches_1.pdf` - Hallucination rates for branch=1
    - `{mode}_api_errors_branches_2_4_8_16.pdf` - 4-panel API error rates
    - `{mode}_api_errors_branch_1.pdf` - API error rates for branch=1
    - `{mode}_length_stops_branches_2_4_8_16.pdf` - 4-panel length stop rates
    - `{mode}_length_stops_branch_1.pdf` - Length stop rates for branch=1


# Real-World Graphs and Proofs Analysis (in /real_world_graphs_and_proofs)

This directory contains the analysis of lookahead and branching distributions in real-world knowledge graphs and mathematical proofs

## Dataset Sources

The analysis includes data from:

**Knowledge Graphs:**
- **ConceptNet 5.7.0** - General knowledge graph
- **ogbl-wikikg2** - Wikidata knowledge graph subset
- **ogbn-papers100M** - Citation network
- **ogb datasets** - Open Graph Benchmark datasets (aggregated)
  - Molecular graphs (molbace, molbbbp, molhiv, etc.)
  - Code graphs (code2)
  - Biological graphs (biokg, proteins)
  - Citation networks (arxiv, citation2, mag)
  - Collaboration networks (collab, ppa)

**Mathematical Proofs:**
- **NaturalProofs** - theorems from three sources:
  - ProofWiki
  - Stacks Project
  - Trench's textbook

All data files are stored in `data/` with Git LFS tracking.

## Generating Histograms

The `generate_histograms.py` script creates a two-row visualization comparing lookahead and branching distributions:

```bash
cd real_world_graphs_and_proofs
python3 generate_histograms.py
```

**Output:** `lookahead_branch_histograms.pdf`

The script automatically:
1. Generates `proof_lookahead_results_full.tsv` from NaturalProofs JSON files (if not already present)
2. Loads and normalizes all lookahead distribution data
3. Loads and normalizes all branching distribution data
4. Creates log-log histograms with quantile lines at [0.5, 0.75, 0.9, 0.99, 0.999]
5. Saves the combined visualization as a PDF

**Data Processing:**
- For individual files: Normalizes to probability distributions
- For directories (e.g., `data/agg_lookahead/`, `data/agg_branches/`): Normalizes each file separately, then averages distributions

## HPC Computation Scripts

For extremely large graphs, use these optimized scripts on an HPC cluster to calculate branching and lookahead metrics:

- **`data_to_triples.py`** - Converts graph datasets to edge triple format for efficient processing
- **`lookahead_numba.py`** - Numba-optimized computation of lookahead distributions
- **`pairwise_distances_numba.py`** - Numba-optimized computation for getting pairwise distance distribution across all source-target pairs

These scripts leverage Numba JIT compilation for high-performance graph metric calculation on large-scale datasets.


# Proof Verification (in /proof_verification)

This directory contains tools for testing LLM ability to detect errors in mathematical proofs. The workflow corrupts proofs from the NaturalProofs dataset and evaluates whether LLMs can identify the introduced errors.

## Workflow Overview

1. **Sample and Corrupt** - Sample proofs stratified by length and introduce single-line errors using LLMs
2. **Verify Proofs** - Run LLM verification on both corrupted proofs (perturbed mode) and clean proofs (trivial mode), and visualize results

## Usage

### Step 1: Sample and Corrupt Proofs

Sample mathematical proofs from NaturalProofs, stratify by length, and introduce errors using two-stage GPT corruption (gpt-4o selects lines, o3-mini corrupts them):

```bash
cd proof_verification

python3 sample_and_corrupt.py \
  --seed 123 \
  --n 100 \
  --bins 5 \
  --hard-bins \
  --min-samples-per-bin 10 \
  --use-gpt \
  --min-lines 5 \
  --out corrupted_proofs.json \
  --cache-file proof_cache.pkl \
  --pick-batch-size 10 \
  --corrupt-batch-size 10
```

**Key Arguments:**
- `--seed SEED` - Random seed for reproducibility
- `--n N` - Number of proof samples to generate
- `--bins N` - Number of length bins for stratified sampling
- `--hard-bins` - Use integer bin boundaries (recommended)
- `--min-samples-per-bin N` - Minimum samples per bin (adaptive binning merges bins if needed)
- `--use-gpt` - Use GPT for two-stage corruption (pick + corrupt)
- `--min-lines N` - Minimum proof length to include
- `--out FILE` - Output JSON file path
- `--cache-file FILE` - LLM response cache file
- `--pick-batch-size N` - Batch size for line selection
- `--corrupt-batch-size N` - Batch size for corruption

**Output:** JSON file containing:
- Sampled proofs with `original_proof` and `modified_proof`
- Metadata: source, theorem info, chosen line, corruption method
- Bin boundaries (if `--hard-bins` used)

### Step 2: Verify Corrupted Proofs (Perturbed Mode)

Run LLM verification on corrupted proofs to test error detection:

```bash
python3 proof_verification.py \
  --input corrupted_proofs.json \
  --out results.json \
  --model gpt-4o \
  --pre-bins \
  --length-bins 5 \
  --min-metric-length 4
```

**Key Arguments:**
- `--input FILE` - Input JSON from sample_and_corrupt.py
- `--out FILE` - Output results JSON path
- `--model {gpt-4o,o3-mini,o3,r1}` - LLM model for verification
- `--pre-bins` - Use bin boundaries from input file
- `--length-bins N` - Number of bins for metrics aggregation
- `--min-metric-length N` - Minimum proof length for metric calculation

**Output:**
- `results.json` - Verification results with per-sample data and aggregated metrics (including full LLM response)
- `results.png` - Accuracy plot by proof length

### Step 3: Verify Clean Proofs (Trivial Mode)

Run verification on the original uncorrupted proofs as a baseline:

```bash
python3 proof_verification.py \
  --input corrupted_proofs.json \
  --out results.json \
  --model gpt-4o \
  --pre-bins \
  --length-bins 5 \
  --min-metric-length 4 \
  --trivial
```

**Output:** `results.trivial.json` - Verification results for clean proofs (should ideally have 100% correct verdicts)

## Data Dependencies

Scripts automatically load NaturalProofs data from:
- `../real_world_graphs_and_proofs/data/naturalproofs_proofwiki.json`
- `../real_world_graphs_and_proofs/data/naturalproofs_stacks.json`
- `../real_world_graphs_and_proofs/data/naturalproofs_trench.json`



