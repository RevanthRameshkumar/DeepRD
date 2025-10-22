# Citation
This repo contains code to perform the experiments and analysis in our paper: "Reasoning Models Reason Well, Until They Don’t"



If you use this data, analysis, or code in your work, please cite:

[bibtex citation coming soon!]
Reasoning Models Reason Well, Until They Don’t
Revanth Rameshkumar, Jimson Huang, Yunxin Sun, Fei Xia, Abulhair Saparov
{revr,fxia}@uw.edu {huan2073,sun1114,asaparov}@purdue.edu



# DeepRD
Evaluation toolset for LLMs using parameterized graph complexity

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
  --no-dedupe                   Disable graph deduplication
  --no-enforce-branches         Disable strict branching enforcement
  -q, --quiet                   Suppress progress output
```

## Quick Start
```bash
# Generate a small test dataset of 5 samples with lookaheads 2,4 and branches 1,2; and write to test.json
python3 graph_generator.py -l 2 4 -b 1 2 --samples 5 -o test.json

# Generate paper's full dataset
python3 graph_generator.py \
  -l 2 3 4 5 6 7 8 9 10 16 32 64 100 128 150 200 250 256 300 350 512 800 \
  -b 2 4 8 16 \
  --samples 10 \
  -o branching_dataset.json

python3 graph_generator.py \
  -l 2 3 4 5 6 7 8 9 10 16 32 64 100 128 150 200 250 256 300 350 512 1024 1536 \
  -b 1 \
  --samples 10 \
  -o trivial_dataset.json
```

## Output Format

Generated JSON contains list of samples with:

```json
{
  "edges": [[1,3], [3,7], ...],      // Directed edge list
  "query": [1, 8],                    // Start and end nodes  
  "lookahead_size": 2,                // Lookahead values
  "max_branches": 2,                  // Branching factor
  "question": "Determine if there..." // Full question text
}
```


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



