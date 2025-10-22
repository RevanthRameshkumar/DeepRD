#!/usr/bin/env python3
"""
Metrics and plotting script for LLM graph reasoning evaluation.

This script analyzes LLM responses on graph path finding tasks,
computing accuracy metrics and generating visualization plots.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import rustworkx as rx


def shortest_path_nodes(record: Dict[str, Any]) -> List[int]:
    """
    Given a record with edges and query, returns the shortest directed path
    from source to target as a list of node labels, or None if no path exists.

    Args:
        record: Dict with "edges" (list of [u,v]) and "query" ([source, target])

    Returns:
        List of node labels forming shortest path, or None if no path exists
    """
    edges = record.get("edges", [])
    src, tgt = record.get("query", [None, None])
    if src is None or tgt is None:
        return None

    # Collect unique labels and validate query nodes
    labels = {u for u, v in edges} | {v for u, v in edges}
    if src not in labels or tgt not in labels:
        return None

    # Build graph and label↔index maps
    G = rx.PyDiGraph()
    label_to_idx = {label: G.add_node(label) for label in labels}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Add directed edges
    for u, v in edges:
        G.add_edge(label_to_idx[u], label_to_idx[v], None)

    # Run Dijkstra for uniform weights
    paths = rx.dijkstra_shortest_paths(
        G,
        label_to_idx[src],
        target=label_to_idx[tgt],
        weight_fn=lambda _: 1.0
    )

    tgt_idx = label_to_idx[tgt]
    if tgt_idx not in paths or not paths[tgt_idx]:
        return None

    # Map index path back to labels
    idx_path = paths[tgt_idx]
    label_path = [idx_to_label[i] for i in idx_path]

    return label_path


def get_usage_token_count(response: Dict[str, Any]) -> int:
    """Extract completion token count from LLM response."""
    if 'usage' in response and 'completion_tokens' in response['usage']:
        return response['usage']['completion_tokens']
    raise ValueError("No completion_tokens in response")


def get_total_token_count(response: Dict[str, Any]) -> int:
    """Extract total token count from LLM response."""
    if 'usage' in response and 'total_tokens' in response['usage']:
        return response['usage']['total_tokens']
    raise ValueError("No total_tokens in response")


def get_input_token_count(response: Dict[str, Any]) -> int:
    """Extract input token count from LLM response."""
    if 'usage' in response and 'prompt_tokens' in response['usage']:
        return response['usage']['prompt_tokens']
    raise ValueError("No prompt_tokens in response")


def is_valid_path(path: List[int], edges: List[List[int]],
                  start_node: int, end_node: int,
                  possible_valid_path: List[int], strict: bool = True) -> int:
    """
    Validate if a path is valid according to the graph edges.

    Returns:
        2: Valid complete path
        0: Invalid path
        -1: Valid first step, but edge hallucination later
        -2: Wrong branch (invalid first step)
    """
    if path is None or path == []:
        return 0

    try:
        if not strict:
            new_path = []
            for p in path:
                try:
                    new_path.append(int(p))
                except (ValueError, TypeError):
                    new_path.append(-111)
            path = new_path
        else:
            path = [int(p) for p in path]
    except (TypeError, ValueError):
        return 0

    if path[0] != start_node:
        return 0
    if path[-1] != end_node:
        return 0
    if len(path) < len(possible_valid_path) * 0.5 and -111 in path:
        return 0

    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))

    for i in range(len(path) - 1):
        if (path[i], path[i + 1]) not in edge_set:
            if not strict:
                if path[i] == -111 or path[i + 1] == -111:
                    continue
            print(f"Missing edge between {path[i]} and {path[i + 1]}")
            if path[1] == possible_valid_path[1]:
                return -1
            else:
                return -2
    return 2


def update_stream_results(current_df: pd.DataFrame, new_record: Dict[str, Any],
                          max_tokens_to_consider: int = -1,
                          skip_length_stopped_generation: bool = False,
                          isolate_branches: List[int] = []) -> pd.DataFrame:
    """
    Update aggregated results dataframe with a new record.

    Args:
        current_df: Existing aggregated dataframe
        new_record: New record to process
        max_tokens_to_consider: Filter records exceeding this token count (-1 = no filter)
        skip_length_stopped_generation: Skip records that hit length limit
        isolate_branches: Only process records with these branch values ([] = all)

    Returns:
        Updated dataframe
    """
    la = new_record["lookahead_size"]
    bs = new_record["max_branches"]
    start_node = new_record["query"][0]
    end_node = new_record["query"][1]
    llm_response = new_record["llm_response"]
    possible_valid_path = shortest_path_nodes(new_record)

    if skip_length_stopped_generation:
        if llm_response['choices'][0]['finish_reason'] == 'length':
            return current_df

    if isolate_branches != []:
        if bs not in isolate_branches:
            return current_df

    if llm_response is not None:
        total_token_count = get_total_token_count(llm_response)
        if total_token_count > max_tokens_to_consider and max_tokens_to_consider != -1:
            print('hit')
            return current_df

    mask = (current_df['lookahead_size'] == la) & (current_df['max_branches'] == bs)

    if current_df[mask].empty:
        new_row = pd.DataFrame([{'lookahead_size': la,
                                 'max_branches': bs,
                                 'total': 0,
                                 'full_valid_path': 0,
                                 'first_step_valid_path': 0,
                                 'usage_token_count': 0,
                                 'edge_hallucination_error': 0,
                                 'max_token_count': 0,
                                 'max_input_token_count': 0,
                                 'wrong_branch': 0,
                                 'length_stop': 0,
                                 'API_error': 0}])
        current_df = pd.concat([current_df, new_row], ignore_index=True)

    mask = (current_df['lookahead_size'] == la) & (current_df['max_branches'] == bs)
    idx = current_df.index[mask][0]

    if llm_response is None:
        current_df.at[idx, 'API_error'] = current_df.at[idx, 'API_error'] + 1
        return current_df

    current_df.at[idx, 'total'] = current_df.at[idx, 'total'] + 1

    extracted_path = new_record.get('extracted_path', None)
    if extracted_path is not None:
        if 'path' in extracted_path:
            extracted_path = extracted_path.get('path', None)
        else:
            extracted_path = extracted_path.get('final_answer', None)

    edges = new_record['edges']
    if 'next_adjective' in new_record:
        valid_path = 2 if extracted_path.strip().lower() == new_record['next_adjective'][0].strip().lower() else 0
    else:
        valid_path = is_valid_path(extracted_path, edges,
                                   start_node, end_node, possible_valid_path, strict=False)

    if valid_path == 2:
        current_df.at[idx, 'full_valid_path'] = current_df.at[idx, 'full_valid_path'] + 1
        current_df.at[idx, 'first_step_valid_path'] = current_df.at[idx, 'first_step_valid_path'] + 1
    elif valid_path == 0:
        pass
    elif valid_path == -1:
        current_df.at[idx, 'edge_hallucination_error'] = current_df.at[idx, 'edge_hallucination_error'] + 1
        current_df.at[idx, 'first_step_valid_path'] = current_df.at[idx, 'first_step_valid_path'] + 1
    elif valid_path == -2:
        current_df.at[idx, 'edge_hallucination_error'] = current_df.at[idx, 'edge_hallucination_error'] + 1
        current_df.at[idx, 'wrong_branch'] = current_df.at[idx, 'wrong_branch'] + 1
    else:
        raise ValueError("Invalid valid_path value")

    tokens_produced = get_usage_token_count(llm_response)
    current_df.at[idx, 'usage_token_count'] = current_df.at[idx, 'usage_token_count'] + tokens_produced

    if tokens_produced > current_df.at[idx, 'max_token_count']:
        current_df.at[idx, 'max_token_count'] = tokens_produced

    input_tokens_used = get_input_token_count(llm_response)
    if input_tokens_used > current_df.at[idx, 'max_input_token_count']:
        current_df.at[idx, 'max_input_token_count'] = input_tokens_used

    if llm_response['choices'][0]['finish_reason'] == 'length':
        current_df.at[idx, 'length_stop'] += 1

    return current_df


def build_counts_df(records: List[Dict[str, Any]],
                   skip_length_stopped_generation: bool,
                   isolate_branches: List[int],
                   max_tokens_to_consider: int = -1) -> pd.DataFrame:
    """
    Build aggregated counts dataframe from list of records.

    Args:
        records: List of evaluation records
        skip_length_stopped_generation: Skip records that hit length limit
        isolate_branches: Only process records with these branch values
        max_tokens_to_consider: Filter records exceeding this token count

    Returns:
        Aggregated dataframe with metrics per (lookahead, max_branches) combination
    """
    df = pd.DataFrame(columns=['lookahead_size', 'max_branches', 'total',
                               'full_valid_path', 'first_step_valid_path',
                               'usage_token_count', 'edge_hallucination_error',
                               'max_token_count', 'max_input_token_count',
                               'wrong_branch', 'length_stop', 'API_error'])
    for rec in records:
        df = update_stream_results(df, rec,
                                  max_tokens_to_consider=max_tokens_to_consider,
                                  skip_length_stopped_generation=skip_length_stopped_generation,
                                  isolate_branches=isolate_branches)

    # Create ratio columns
    df['ratio'] = 0.0
    mask = df['total'] != 0
    df.loc[mask, 'ratio'] = df.loc[mask, 'full_valid_path'] / df.loc[mask, 'total']
    df.loc[mask, 'first_step_ratio'] = df.loc[mask, 'first_step_valid_path'] / df.loc[mask, 'total']
    df.loc[mask, 'avg_tokens'] = df.loc[mask, 'usage_token_count'] / df.loc[mask, 'total']
    df.loc[mask, 'API_error_rate'] = df.loc[mask, 'API_error'] / (df.loc[mask, 'total'] + df.loc[mask, 'API_error'])
    df.loc[mask, 'length_stop_rate'] = df.loc[mask, 'length_stop'] / df.loc[mask, 'total']
    df.loc[mask, 'edge_hallucination_error_rate'] = df.loc[mask, 'edge_hallucination_error'] / df.loc[mask, 'total']

    return df.sort_values('lookahead_size')


def plot_four_branches(model_dfs: Dict[str, pd.DataFrame], y_label: str,
                      branches: Tuple[int, ...] = (2, 4, 8, 16),
                      save_path: str = None):
    """
    Create 4-panel plot showing accuracy for different branch values.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        y_label: Y-axis label
        branches: Tuple of 4 branch values to plot
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharey=True)
    axes = axes.flatten()

    for idx, (ax, mb) in enumerate(zip(axes, branches)):
        letter = chr(ord('a') + idx)
        for model_name, df in model_dfs.items():
            sub = df[df['max_branches'] == mb]
            if sub.empty:
                continue
            ax.plot(sub['lookahead_size'], sub['ratio'], marker='o', label=model_name)

        ax.annotate(f'({letter})', xy=(-0.05, 1.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=14, fontweight='bold')

        ax.set_title(f'Number of Branches (B) = {mb}', loc='center', fontsize=14)
        ax.set_xlabel('Lookahead (L)')
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')

    return fig


def plot_four_branches_first_node(model_dfs: Dict[str, pd.DataFrame],
                                  branches: Tuple[int, ...] = (2, 4, 8, 16),
                                  save_path: str = None):
    """
    Create 4-panel plot showing first-step accuracy for different branch values.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        branches: Tuple of 4 branch values to plot
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharey=True)
    axes = axes.flatten()

    for idx, (ax, mb) in enumerate(zip(axes, branches)):
        letter = chr(ord('a') + idx)

        for model_name, df in model_dfs.items():
            sub = df[df['max_branches'] == mb]
            if sub.empty:
                continue
            ax.plot(sub['lookahead_size'], sub['first_step_ratio'], marker='o', label=model_name)

        ax.annotate(f'({letter})', xy=(-0.05, 1.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=14, fontweight='bold')

        ax.set_title(f'Number of Branches (B) = {mb}', loc='center', fontsize=14)
        ax.set_xlabel('Lookahead (L)')
        ax.set_ylabel('Next Step Accuracy')
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_branch_one_first_node(model_dfs: Dict[str, pd.DataFrame],
                               x_label: str, y_label: str,
                               save_path: str = None):
    """
    Create plot showing first-step accuracy for branch=1 across all models.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(6, 4))

    for model_name, df in model_dfs.items():
        sub = df[df['max_branches'] == 1]
        if sub.empty:
            continue
        plt.plot(sub['lookahead_size'], sub['first_step_ratio'], marker='o', label=model_name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('B = 1')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.ylim(0.0, 1.0)

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    return plt.gcf()


def plot_four_branches_tokens(model_dfs: Dict[str, pd.DataFrame],
                              branches: Tuple[int, ...] = (2, 4, 8, 16),
                              save_path: str = None):
    """
    Create 4-panel plot showing token usage for different branch values.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        branches: Tuple of 4 branch values to plot
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharey=True)
    axes = axes.flatten()

    for idx, (ax, mb) in enumerate(zip(axes, branches)):
        letter = chr(ord('a') + idx)

        for model_name, df in model_dfs.items():
            sub = df[df['max_branches'] == mb]
            if sub.empty:
                continue
            ax.plot(sub['lookahead_size'], sub['avg_tokens'], marker='o', label=model_name)

        ax.annotate(f'({letter})', xy=(-0.05, 1.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=14, fontweight='bold')

        ax.set_title(f'Number of Branches (B) = {mb}', loc='center', fontsize=14)
        ax.set_xlabel('Lookahead (L)')
        ax.set_ylabel('Avg Completion Tokens')
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_single_branch_tokens(model_dfs: Dict[str, pd.DataFrame],
                              branch: int = 8,
                              save_path: str = None):
    """
    Create plot showing token usage for a single branch value.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        branch: Branch value to plot
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(6, 4))
    for model_name, df in model_dfs.items():
        sub = df[df['max_branches'] == branch]
        if sub.empty:
            continue
        plt.plot(sub['lookahead_size'], sub['avg_tokens'], marker='o', label=model_name)
    plt.xlabel('Graph Depth')
    plt.ylabel('Avg Completion Tokens')
    plt.title(f'B = {branch}')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    return plt.gcf()


def plot_branch_one(model_dfs: Dict[str, pd.DataFrame],
                   x_label: str, y_label: str,
                   save_path: str = None):
    """
    Create plot showing accuracy for branch=1 across all models.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(6, 4))
    for model_name, df in model_dfs.items():
        sub = df[df['max_branches'] == 1]
        if sub.empty:
            continue
        plt.plot(sub['lookahead_size'], sub['ratio'], marker='o', label=model_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('B = 1')
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    return plt.gcf()


def plot_edge_hallucinations(model_dfs: Dict[str, pd.DataFrame],
                             x_label: str, y_label: str,
                             branches: Tuple[int, ...] = (2, 4, 8, 16),
                             save_path: str = None):
    """
    Create 4-panel plot showing edge hallucination rates for different branch values.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        branches: Tuple of 4 branch values to plot
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharey=True)
    axes = axes.flatten()

    for idx, (ax, mb) in enumerate(zip(axes, branches)):
        letter = chr(ord('a') + idx)

        for model_name, df in model_dfs.items():
            sub = df[df['max_branches'] == mb]
            if sub.empty:
                continue

            edge_hallucinations_rate = sub['edge_hallucination_error_rate']

            ax.plot(sub['lookahead_size'], edge_hallucinations_rate, marker='o', label=model_name)

        ax.annotate(f'({letter})', xy=(-0.05, 1.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=14, fontweight='bold')

        ax.set_title(f'Number of Branches (B) = {mb}', loc='center', fontsize=14)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Hallucination Rate')
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')

    return fig


def plot_edge_hallucinations_1b(model_dfs: Dict[str, pd.DataFrame],
                                x_label: str, y_label: str,
                                save_path: str = None):
    """
    Create plot showing edge hallucination rates for branch=1 across all models.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(6, 4))

    for model_name, df in model_dfs.items():
        sub = df[df['max_branches'] == 1]
        if sub.empty:
            continue

        edge_hallucinations_count = sub['edge_hallucination_error']

        plt.plot(sub['lookahead_size'], edge_hallucinations_count, marker='o', label=model_name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('B = 1')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.ylim(0.0, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    return plt.gcf()


def plot_api_errors(model_dfs: Dict[str, pd.DataFrame],
                   x_label: str, y_label: str,
                   branches: Tuple[int, ...] = (2, 4, 8, 16),
                   save_path: str = None):
    """
    Create 4-panel plot showing API error rates for different branch values.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        branches: Tuple of 4 branch values to plot
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharey=True)
    axes = axes.flatten()

    for idx, (ax, mb) in enumerate(zip(axes, branches)):
        letter = chr(ord('a') + idx)

        for model_name, df in model_dfs.items():
            sub = df[df['max_branches'] == mb]
            if sub.empty:
                continue

            api_error_count = sub['API_error_rate']

            ax.plot(sub['lookahead_size'], api_error_count, marker='o', label=model_name)

        ax.annotate(f'({letter})', xy=(-0.05, 1.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=14, fontweight='bold')

        ax.set_title(f'Number of Branches (B) = {mb}', loc='center', fontsize=14)
        ax.set_xlabel(x_label)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel('API Error Rate')
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')

    return fig


def plot_api_errors_1b(model_dfs: Dict[str, pd.DataFrame],
                      x_label: str, y_label: str,
                      save_path: str = None):
    """
    Create plot showing API error rates for branch=1 across all models.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(6, 4))

    for model_name, df in model_dfs.items():
        sub = df[df['max_branches'] == 1]
        if sub.empty:
            continue

        api_error_rate = sub['API_error_rate']

        plt.plot(sub['lookahead_size'], api_error_rate, marker='o', label=model_name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('B = 1')
    plt.legend()
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    return plt.gcf()


def plot_length_stops(model_dfs: Dict[str, pd.DataFrame],
                     x_label: str, y_label: str,
                     branches: Tuple[int, ...] = (2, 4, 8, 16),
                     save_path: str = None):
    """
    Create 4-panel plot showing length stop rates for different branch values.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        branches: Tuple of 4 branch values to plot
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(6, 10), sharey=True)
    axes = axes.flatten()

    for idx, (ax, mb) in enumerate(zip(axes, branches)):
        letter = chr(ord('a') + idx)

        for model_name, df in model_dfs.items():
            sub = df[df['max_branches'] == mb]
            if sub.empty:
                continue

            length_stop_rate = sub['length_stop_rate']

            ax.plot(sub['lookahead_size'], length_stop_rate, marker='o', label=model_name)

        ax.annotate(f'({letter})', xy=(-0.05, 1.03), xycoords='axes fraction',
                    ha='right', va='bottom', fontsize=14, fontweight='bold')

        ax.set_title(f'Number of Branches (B) = {mb}', loc='center', fontsize=14)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Length Stop Rate')
        ax.grid(alpha=0.2)

        ax.set_ylim(0.0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, format='pdf', bbox_inches='tight')

    return fig


def plot_length_stops_1b(model_dfs: Dict[str, pd.DataFrame],
                        x_label: str, y_label: str,
                        save_path: str = None):
    """
    Create plot showing length stop rates for branch=1 across all models.

    Args:
        model_dfs: Dict mapping model_name -> dataframe
        x_label: X-axis label
        y_label: Y-axis label
        save_path: Path to save figure (PDF format)

    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(6, 4))

    for model_name, df in model_dfs.items():
        sub = df[df['max_branches'] == 1]
        if sub.empty:
            continue

        length_stop_rate = sub['length_stop_rate']

        plt.plot(sub['lookahead_size'], length_stop_rate, marker='o', label=model_name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('B = 1')
    plt.legend()
    plt.grid(alpha=0.2)

    ax = plt.gca()
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')

    return plt.gcf()


def make_all_figures_and_tables(data_file_list: List[Tuple[str, str]],
                               prepend_str: str,
                               accuracy_label: str,
                               output_dir: str = 'output'):
    """
    Generate all figures and tables from evaluation data.

    Args:
        data_file_list: List of (model_name, file_path) tuples
        prepend_str: Prefix for output files ('symbolic' or 'logic')
        accuracy_label: Y-axis label for accuracy plots
        output_dir: Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    model_dfs = {}
    for model_name, path in data_file_list:
        print(f"\nLoading data for {model_name} from {path}")
        with open(path, 'r') as f:
            streamed_data = json.load(f)

        model_dfs[model_name] = build_counts_df(streamed_data,
                                                skip_length_stopped_generation=False,
                                                isolate_branches=[],
                                                max_tokens_to_consider=-1)

        print(f"\nAggregated counts for {model_name}:")
        tsv_path = os.path.join(output_dir, f'{prepend_str}_{model_name}_metrics.tsv')
        model_dfs[model_name].to_csv(tsv_path, sep='\t', index=False)
        print(f"  Saved metrics to: {tsv_path}")
        print(f"  Total samples processed: {model_dfs[model_name]['total'].sum()}")

        # Verify expected sample count (10 samples per lookahead/branch combo)
        expected = len(model_dfs[model_name]) * 10
        actual = model_dfs[model_name]['total'].sum() + model_dfs[model_name]['API_error'].sum()
        assert expected == actual, f"Sample count mismatch: expected {expected}, got {actual}"

    # Generate all plots
    print(f"\nGenerating plots...")

    plot_four_branches(
        model_dfs,
        y_label=accuracy_label,
        branches=(2, 4, 8, 16),
        save_path=os.path.join(output_dir, f'{prepend_str}_full_path_accuracy_branches_2_4_8_16.pdf')
    )
    print(f"  ✓ {prepend_str}_full_path_accuracy_branches_2_4_8_16.pdf")

    plot_four_branches_first_node(
        model_dfs,
        branches=(2, 4, 8, 16),
        save_path=os.path.join(output_dir, f'{prepend_str}_first_node_accuracy_branches_2_4_8_16.pdf')
    )
    print(f"  ✓ {prepend_str}_first_node_accuracy_branches_2_4_8_16.pdf")

    plot_branch_one_first_node(
        model_dfs,
        "Graph Depth",
        "Next Step Accuracy",
        save_path=os.path.join(output_dir, f'{prepend_str}_first_node_accuracy_branches_1.pdf')
    )
    print(f"  ✓ {prepend_str}_first_node_accuracy_branches_1.pdf")

    plot_four_branches_tokens(
        model_dfs,
        branches=(2, 4, 8, 16),
        save_path=os.path.join(output_dir, f'{prepend_str}_avg_tokens_branches_2_4_8_16.pdf')
    )
    print(f"  ✓ {prepend_str}_avg_tokens_branches_2_4_8_16.pdf")

    plot_single_branch_tokens(
        model_dfs,
        branch=8,
        save_path=os.path.join(output_dir, f'{prepend_str}_avg_tokens_branches_8.pdf')
    )
    print(f"  ✓ {prepend_str}_avg_tokens_branches_8.pdf")

    plot_single_branch_tokens(
        model_dfs,
        branch=1,
        save_path=os.path.join(output_dir, f'{prepend_str}_avg_tokens_branches_1.pdf')
    )
    print(f"  ✓ {prepend_str}_avg_tokens_branches_1.pdf")

    plot_branch_one(
        model_dfs,
        'Graph Depth',
        accuracy_label,
        save_path=os.path.join(output_dir, f'{prepend_str}_valid_path_ratio_branch_1.pdf')
    )
    print(f"  ✓ {prepend_str}_valid_path_ratio_branch_1.pdf")

    plot_edge_hallucinations(
        model_dfs,
        'Lookahead',
        y_label="Hallucination Rate",
        save_path=os.path.join(output_dir, f'{prepend_str}_edge_hallucinations_branches_2_4_8_16.pdf')
    )
    print(f"  ✓ {prepend_str}_edge_hallucinations_branches_2_4_8_16.pdf")

    plot_edge_hallucinations_1b(
        model_dfs,
        'Graph Depth',
        y_label="Hallucination Rate",
        save_path=os.path.join(output_dir, f'{prepend_str}_edge_hallucinations_branches_1.pdf')
    )
    print(f"  ✓ {prepend_str}_edge_hallucinations_branches_1.pdf")

    plot_api_errors(
        model_dfs,
        'Lookahead',
        'API Error Rate',
        save_path=os.path.join(output_dir, f'{prepend_str}_api_errors_branches_2_4_8_16.pdf')
    )
    print(f"  ✓ {prepend_str}_api_errors_branches_2_4_8_16.pdf")

    plot_api_errors_1b(
        model_dfs,
        'Graph Depth',
        'API Error Rate',
        save_path=os.path.join(output_dir, f'{prepend_str}_api_errors_branch_1.pdf')
    )
    print(f"  ✓ {prepend_str}_api_errors_branch_1.pdf")

    plot_length_stops(
        model_dfs,
        'Lookahead',
        'Length Stop Rate',
        save_path=os.path.join(output_dir, f'{prepend_str}_length_stops_branches_2_4_8_16.pdf')
    )
    print(f"  ✓ {prepend_str}_length_stops_branches_2_4_8_16.pdf")

    plot_length_stops_1b(
        model_dfs,
        'Graph Depth',
        'Length Stop Rate',
        save_path=os.path.join(output_dir, f'{prepend_str}_length_stops_branch_1.pdf')
    )
    print(f"  ✓ {prepend_str}_length_stops_branch_1.pdf")

    print(f"\n✓ All outputs saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate metrics and plots for LLM graph reasoning evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze symbolic reasoning results
  python3 llm_results/llm_metrics.py \\
    --files llm_results/FINAL_llm_results_*_R1.json llm_results/FINAL_llm_results_*_o3.json \\
    --models R1 o3 \\
    --mode symbolic \\
    --output-dir output/symbolic

  # Analyze logic reasoning results
  python3 llm_results/llm_metrics.py \\
    --files llm_results/FINAL_LOGIC_llm_results_*_V3.json \\
    --models V3 \\
    --mode logic \\
    --output-dir output/logic

Output:
  - TSV files: {mode}_{model}_metrics.tsv (one per model)
  - PDF plots: Full path accuracy, branch=1 accuracy, token usage
  - PNG plots: First-step accuracy, 4-branch token usage
        """
    )

    parser.add_argument(
        '--files', '-f',
        nargs='+',
        required=True,
        help='List of result JSON files to analyze'
    )

    parser.add_argument(
        '--models', '-m',
        nargs='+',
        required=True,
        help='List of model names (must match order and count of --files)'
    )

    parser.add_argument(
        '--mode',
        choices=['symbolic', 'logic'],
        required=True,
        help='Evaluation mode: symbolic (graph path) or logic (next step)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='output',
        help='Output directory for plots and metrics (default: output)'
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.files) != len(args.models):
        parser.error(f"Number of files ({len(args.files)}) must match number of models ({len(args.models)})")

    # Check all files exist
    for file_path in args.files:
        if not os.path.exists(file_path):
            parser.error(f"File not found: {file_path}")

    # Set accuracy label based on mode
    accuracy_label = 'Full Path Accuracy' if args.mode == 'symbolic' else 'Next Step Accuracy'

    # Build data file list
    data_file_list = list(zip(args.models, args.files))

    print(f"Mode: {args.mode}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Files: {len(args.files)}")
    print(f"Output directory: {args.output_dir}")

    # Generate all figures and tables
    make_all_figures_and_tables(
        data_file_list,
        prepend_str=args.mode,
        accuracy_label=accuracy_label,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
