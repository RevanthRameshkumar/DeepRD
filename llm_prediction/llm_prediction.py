#!/usr/bin/env python3
"""
Unified LLM prediction script for symbolic and logic reasoning tasks.

This script combines the functionality of saparov_inference.py and
saparov_inference_logic.py into a single CLI tool.
"""

import argparse
import json
import datetime
from copy import deepcopy
from llm_inference import CachingLLM, get_content_from_response, get_path_extraction_content_from_response, sanitize_filename


def get_model_config(model_name, mode, cache_dir):
    """
    Get model configuration based on model name and mode.

    Returns:
        tuple: (inference_llm, gpt4o_mini_llm, output_prefix)
    """
    # Cache file paths
    if mode == 'symbolic':
        together_ai_cache = f"{cache_dir}/together_ai_llm_cache.pkl"
        openai_cache = f"{cache_dir}/openai_ai_llm_cache.pkl"
        evaluation_cache = f"{cache_dir}/evaluation_cache.pkl"
        output_prefix = "FINAL_llm_results"
    else:  # logic mode
        together_ai_cache = f"{cache_dir}/logic_together_ai_llm_cache.pkl"
        openai_cache = f"{cache_dir}/logic_openai_ai_llm_cache_2.pkl"
        evaluation_cache = f"{cache_dir}/logic_evaluation_cache2.pkl"
        output_prefix = "FINAL_LOGIC_llm_results"

    # Initial messages based on mode
    if mode == 'symbolic':
        inference_initial_message = [
            {"role": "system",
             "content": 'When answering a yes or no question, always start the answer with |YES| or |NO| depending on your answer'}
        ]
        evaluation_initial_message = [
            {"role": "system",
             "content": 'Given the input text, respond ONLY with the path extracted from the text in the json format {"path":[...]}'},
        ]
    else:  # logic mode
        inference_initial_message = [
            {"role": "system",
             "content": 'For the final answer, only respond with the attribute that fills in the blank, example: "Final Answer: pink"'}
        ]
        evaluation_initial_message = [
            {"role": "system",
             "content": 'Given the input text, respond ONLY with the next attribute extracted in JSON form like {"final_answer": "pink"}'},
        ]

    # GPT-4o-mini for extraction (same for both modes)
    gpt4o_mini_llm = CachingLLM(
        model_params={"model": "gpt-4o-mini", "temperature": 0.0},
        initial_messages=evaluation_initial_message,
        cache_file=evaluation_cache
    )

    # Model-specific configurations
    if model_name == 'r1':
        inference_llm = CachingLLM(
            model_params={"model": "deepseek-ai/DeepSeek-R1", "temperature": 0.0, "max_tokens": 80000},
            initial_messages=inference_initial_message,
            cache_file=together_ai_cache
        )
    elif model_name == 'v3':
        inference_llm = CachingLLM(
            model_params={"model": "deepseek-ai/DeepSeek-V3", "temperature": 0.0},
            initial_messages=inference_initial_message,
            cache_file=together_ai_cache
        )
    elif model_name == 'o3-mini':
        inference_llm = CachingLLM(
            model_params={"model": "o3-mini", "reasoning_effort": args.reasoning_effort},
            initial_messages=inference_initial_message,
            cache_file=openai_cache
        )
    elif model_name == 'o3':
        inference_llm = CachingLLM(
            model_params={"model": "o3", "reasoning_effort": args.reasoning_effort},
            initial_messages=inference_initial_message,
            cache_file=openai_cache
        )
    elif model_name == 'gpt4o':
        inference_llm = CachingLLM(
            model_params={"model": "gpt-4o", "temperature": 0.0},
            initial_messages=inference_initial_message,
            cache_file=openai_cache
        )
    elif model_name == 'gpt4':
        inference_llm = CachingLLM(
            model_params={"model": "gpt-4", "temperature": 0.0},
            initial_messages=inference_initial_message,
            cache_file=openai_cache
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return inference_llm, gpt4o_mini_llm, output_prefix


def prepare_messages(data_list, mode):
    """
    Prepare messages for inference based on mode.

    For symbolic mode: uses item['question'] directly
    For logic mode: transforms predicates and combines with logical_question
    """
    messages = []

    if mode == 'symbolic':
        for item in data_list:
            messages.append(item['question'])
    else:  # logic mode
        for item in data_list:
            new_predicates = item['logic_predicates'].replace(
                'Given the following list of predicates:\n',
                'Suppose we have the following list of facts:\n'
            )
            new_message = new_predicates + '\n' + item['logical_question']
            messages.append(new_message)

    return messages


def main(args):
    # Load input data
    print(f"Loading data from {args.input_file}...")
    with open(args.input_file, 'r') as f:
        data_list = json.load(f)
    print(f'Total data size: {len(data_list)}')

    # Get model configuration
    inference_model, gpt4o_mini_llm, output_prefix = get_model_config(
        args.model, args.mode, args.cache_dir
    )

    # Create output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = sanitize_filename(
        f"{output_prefix}_{timestamp}_{inference_model.model_params['model']}.json"
    )
    output_path = f"{args.output_dir}/{output_filename}"
    print(f'Output will be saved to: {output_path}')

    # Prepare messages for inference
    sample_items = data_list
    print(f'Working on data of length: {len(sample_items)}')

    messages = prepare_messages(sample_items, args.mode)

    # Stage 1: Generate responses with main model
    print(f"\nStage 1: Generating responses with {inference_model.model_params['model']}...")
    responses = inference_model.generate_responses(messages)

    # Collect results
    results = []
    for question_text, llm_response, item in zip(messages, responses, sample_items):
        new_item = deepcopy(item)
        new_item["question"] = question_text
        new_item["llm_response"] = llm_response
        new_item["lookahead_size"] = item.get('lookahead_size', -1)
        results.append(new_item)

    # Clean results (remove None responses)
    print(f'Count before cleaning: {len(results)}')
    cleaned_results = []
    for item in results:
        if item['llm_response'] is not None:
            item['llm_response'] = item['llm_response'].to_dict()
            cleaned_results.append(item)
    print(f'Count after cleaning: {len(cleaned_results)}')
    results = cleaned_results

    # Stage 2: Extract paths/answers with GPT-4o-mini
    print(f"\nStage 2: Extracting paths with gpt-4o-mini...")
    updated_json = []

    messages = []
    for item in results:
        content = get_path_extraction_content_from_response(item['llm_response'])
        messages.append(content)
    responses = gpt4o_mini_llm.generate_responses(messages)

    for item, llm_response in zip(results, responses):
        try:
            json_path = json.loads(get_content_from_response(llm_response.to_dict()))
        except (TypeError, json.JSONDecodeError):
            json_path = None

        item['extracted_path'] = json_path
        updated_json.append(item)

    # Write results to file
    with open(output_path, 'w') as outfile:
        json.dump(updated_json, outfile, indent=2)

    print(f"\n✓ Results successfully written to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run LLM inference for symbolic or logic reasoning tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Symbolic reasoning with o3-mini
  python llm_prediction.py --mode symbolic --model o3-mini \\
    --input-file ../graph_generator/output.json \\
    --reasoning-effort medium

  # Logic reasoning with DeepSeek R1
  python llm_prediction.py --mode logic --model r1 \\
    --input-file ../graph_generator/logic_dataset.json
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['symbolic', 'logic'],
                        help='Reasoning mode: symbolic (graph path finding) or logic (attribute prediction)')

    parser.add_argument('--model', type=str, required=True,
                        choices=['o3', 'o3-mini', 'r1', 'v3', 'gpt4o', 'gpt4'],
                        help='Model to use for inference')

    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to input JSON file (generated by graph_generator.py)')

    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save output files (default: current directory)')

    parser.add_argument('--cache-dir', type=str, default='.',
                        help='Directory for cache files (default: current directory)')

    parser.add_argument('--reasoning-effort', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort for o3/o3-mini models (default: medium)')

    args = parser.parse_args()

    main(args)
