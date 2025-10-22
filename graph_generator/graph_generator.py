#!/usr/bin/env python3
"""
DeepRD Graph Dataset Generator

Generates parameterized graph datasets for testing reasoning models.
Based on the paper's methodology with configurable lookahead and branching parameters.

Citation: Based on work from https://github.com/asaparov/learning_to_search
"""

import argparse
import json
import random
import sys
import time
import datetime
import subprocess
from collections import Counter
from importlib.util import find_spec
from os import system
from os.path import getmtime
from typing import Iterable, Hashable

import networkx as nx
import numpy as np


# ============================================================================
# Graph Utility Functions
# ============================================================================

def same_structure(edges1, edges2, directed=True):
    """Check if two edge lists define isomorphic graphs."""
    G1 = nx.DiGraph() if directed else nx.Graph()
    G2 = nx.DiGraph() if directed else nx.Graph()
    G1.add_edges_from(edges1)
    G2.add_edges_from(edges2)
    return nx.faster_could_be_isomorphic(G1, G2)


def degree_map(edges):
    """Calculate out-degree for each node in a directed graph."""
    deg = Counter()
    for u, v in edges:
        deg[u] += 1
    return deg


# ============================================================================
# C++ Module Compilation
# ============================================================================

def build_module(name):
    """Compile C++ extension module using pybind11."""
    if sys.platform.startswith("win"):
        try:
            includes = subprocess.check_output(
                ["python", "-m", "pybind11", "--includes"]
            ).decode().strip()
        except subprocess.CalledProcessError:
            print("ERROR: Could not get pybind11 include flags.")
            sys.exit(1)
        msvc_includes = includes.replace("-I", "/I")
        cmd = f"cl /LD /O2 /DNDEBUG /MD {msvc_includes} /I. {name}.cpp /Fe:{name}.pyd"
    else:
        cmd = (
            f"g++ -Ofast -DNDEBUG -fno-stack-protector -Wall -Wpedantic "
            f"-shared -fPIC $(python3 -m pybind11 --includes) -I. {name}.cpp "
            f"-o {name}$(python3-config --extension-suffix)"
        )

    print(f"Compiling C++ module: {name}")
    if system(cmd) != 0:
        print(f"ERROR: Unable to compile `{name}.cpp`.")
        sys.exit(1)


def load_graph_generation_module():
    """Load or compile the C++ graph generation module."""
    import importlib
    global generator

    try:
        generator_spec = find_spec('generator')
        if generator_spec is None:
            raise ModuleNotFoundError
        elif getmtime(generator_spec.origin) < getmtime('generator.cpp'):
            print("C++ module `generator` is out-of-date. Recompiling...")
            build_module("generator")
        generator = importlib.import_module('generator')
    except ModuleNotFoundError:
        print("C++ module `generator` not found. Compiling from source...")
        build_module("generator")
        generator = importlib.import_module('generator')

    print("C++ module `generator` loaded successfully.")
    return generator


# ============================================================================
# Graph Generation
# ============================================================================

def list_to_edge_list(nums, QUERY_PREFIX_TOKEN, PADDING_TOKEN, EDGE_PREFIX_TOKEN):
    """Convert tokenized representation to edge list and query."""
    if QUERY_PREFIX_TOKEN not in nums:
        raise ValueError("Query prefix token not found")

    query_index = np.where(nums == QUERY_PREFIX_TOKEN)[0][0]
    query = (int(nums[query_index + 1]), int(nums[query_index + 2]))
    nums = nums[:query_index]

    nums = [n for n in nums if n != PADDING_TOKEN]

    groups = []
    current_group = []
    for n in nums:
        if n == EDGE_PREFIX_TOKEN:
            if current_group:
                groups.append(current_group)
            current_group = []
        else:
            current_group.append(n)
    if current_group:
        groups.append(current_group)

    edge_list = []
    for group in groups:
        if len(group) >= 2:
            edge_list.append((int(group[0]), int(group[1])))

    return edge_list, query


def generate_examples(lookahead_size, num_examples, max_branches,
                     dedupe_graphs, enforce_branch_size, random_seed=9, verbose=True):
    """
    Generate graph examples with specified parameters.

    Args:
        lookahead_size: Minimum path length for generated graphs
        num_examples: Number of unique graphs to generate
        max_branches: Number of outgoing edges from start node
        dedupe_graphs: Remove isomorphic duplicates
        enforce_branch_size: Enforce exact branching factor at start node
        random_seed: Seed for reproducibility (default: 9, same as paper)
        verbose: Print generation progress

    Returns:
        List of (edge_list, query) tuples
    """
    random.seed(random_seed)
    previous_edge_lists = []
    final_graphs = []
    start_time = time.perf_counter()

    while len(final_graphs) < num_examples:
        if verbose:
            print(f"Generated: {len(final_graphs)}/{num_examples}", end='\r')

        generator.set_seed(random.randint(1, 500))

        dataset_size = 10
        max_lookahead = lookahead_size
        max_input_size = 2 * ((max_branches * max_lookahead) + 1) * 3 + 5

        QUERY_PREFIX_TOKEN = (max_input_size - 5) // 3 + 4
        PADDING_TOKEN = (max_input_size - 5) // 3 + 3
        EDGE_PREFIX_TOKEN = (max_input_size - 5) // 3 + 2

        if max_branches > 1:
            output = generator.generate_training_set(
                max_input_size, dataset_size, max_lookahead,
                (max_input_size - 5) // 3, set(), 1, max_input_size,
                max_branches, False
            )
            edge_list_and_query = output[0]
        elif max_branches == 1:
            edge_list_and_query = []
            for i in range(dataset_size):
                nodes = list(range(1, max_lookahead + 2))
                random.shuffle(nodes)
                edges = [nodes[e:e+2] for e in range(len(nodes) - 1)]
                random.shuffle(edges)

                nums = [PADDING_TOKEN]
                for e in edges:
                    nums.extend([EDGE_PREFIX_TOKEN, e[0], e[1]])
                nums.extend([QUERY_PREFIX_TOKEN, nodes[0], nodes[-1]])
                edge_list_and_query.append(nums)
            edge_list_and_query = np.array(edge_list_and_query)
        else:
            raise ValueError("max_branches must be >= 1")

        for i in range(edge_list_and_query.shape[0]):
            if len(final_graphs) >= num_examples:
                break

            edge_list, query = list_to_edge_list(
                edge_list_and_query[i],
                QUERY_PREFIX_TOKEN, PADDING_TOKEN, EDGE_PREFIX_TOKEN
            )

            degrees = degree_map(edge_list)

            # Enforce branching factor constraint
            if enforce_branch_size and degrees[query[0]] != max_branches:
                continue

            # Deduplicate graphs
            if dedupe_graphs and max_branches != 1:
                # Skip if graph has only simple structure
                skip_graph = all(
                    v <= 2 for k, v in degrees.items() if k != query[0]
                )
                if skip_graph:
                    continue

                # Skip if isomorphic to previous graph
                if any(same_structure(edge_list, prev) for prev in previous_edge_lists):
                    continue

            previous_edge_lists.append(edge_list)
            final_graphs.append((edge_list, query))

    if verbose:
        elapsed = time.perf_counter() - start_time
        print(f"\nGenerated {num_examples} graphs in {elapsed:.2f}s "
              f"({num_examples/elapsed:.1f} graphs/s)")

    return final_graphs


def extract_data_points(edge_and_query_list, lookahead_size, max_branches):
    """
    Convert graph representations to dataset format.

    Args:
        edge_and_query_list: List of (edge_list, query) tuples
        lookahead_size: Lookahead parameter for metadata
        max_branches: Branch parameter for metadata

    Returns:
        List of data point dictionaries
    """
    previously_seen_graph = set()
    data_points = []

    for edge_list, query in edge_and_query_list:
        graph_key = str(edge_list)

        # Skip duplicate graphs
        if graph_key in previously_seen_graph and max_branches != 1:
            continue
        previously_seen_graph.add(graph_key)

        # Shuffle edges for presentation variety
        random.shuffle(edge_list)
        deg = degree_map(edge_list)

        point = {
            'edges': edge_list,
            'query': query,
            'lookahead_size': lookahead_size,
            'max_branches': deg[query[0]],
            'question': (
                f"Determine if there is a path between two nodes in the graph. "
                f"Note that (i,j) means that node i and node j are connected "
                f"with a directed edge.\n\t Graph: {graph_key}\n\t "
                f"Q: Is there a path between node {query[0]} and node {query[1]} "
                f"and what is the path?\nA:"
            )
        }
        data_points.append(point)

    return data_points


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate graph datasets for reasoning model evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # write to test.json
    %(prog)s graph_generator.py -l 2 4 -b 1 2 --samples 5 -o test.json

    # Generate paper's full dataset
    %(prog)s graph_generator.py \
    -l 2 3 4 5 6 7 8 9 10 16 32 64 100 128 150 200 250 256 300 350 512 800 \
    -b 2 4 8 16 \
    --samples 10 \
    -o branching_dataset.json

    %(prog)s graph_generator.py \
    -l 2 3 4 5 6 7 8 9 10 16 32 64 100 128 150 200 250 256 300 350 512 1024 1536 \
    -b 1 \
    --samples 10 \
    -o trivial_dataset.json
        """
    )

    parser.add_argument(
        '--lookaheads', '-l',
        type=int, nargs='+', required=True,
        help='Lookahead values'
    )
    parser.add_argument(
        '--branches', '-b',
        type=int, nargs='+', required=True,
        help='Branch values (number of outgoing edges from start node)'
    )
    parser.add_argument(
        '--samples', '-n',
        type=int, default=10,
        help='Number of samples per (lookahead, branch) combination (default: 10)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str, default=None,
        help='Output filename (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--seed',
        type=int, default=9,
        help='Random seed for reproducibility (default: 9, same as paper)'
    )
    parser.add_argument(
        '--no-dedupe',
        action='store_true',
        help='Disable graph deduplication (allow isomorphic duplicates)'
    )
    parser.add_argument(
        '--no-enforce-branches',
        action='store_true',
        help='Disable strict branching factor enforcement'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Load C++ module
    load_graph_generation_module()

    # Configuration
    lookaheads = args.lookaheads
    branches = args.branches
    sample_size = args.samples
    dedupe_graphs = not args.no_dedupe
    enforce_branch_size = not args.no_enforce_branches
    verbose = not args.quiet

    if verbose:
        print(f"\nDataset Configuration:")
        print(f"  Lookaheads: {lookaheads}")
        print(f"  Branches: {branches}")
        print(f"  Samples per combination: {sample_size}")
        print(f"  Total combinations: {len(lookaheads) * len(branches)}")
        print(f"  Expected total samples: {len(lookaheads) * len(branches) * sample_size}")
        print(f"  Random seed: {args.seed}")
        print(f"  Deduplication: {dedupe_graphs}")
        print(f"  Enforce branching: {enforce_branch_size}\n")

    all_data_points = []
    total_combinations = len(lookaheads) * len(branches)
    current_combination = 0

    for lookahead in lookaheads:
        for branch_size in branches:
            current_combination += 1
            if verbose:
                print(f"\n[{current_combination}/{total_combinations}] "
                      f"Generating: lookahead={lookahead}, branches={branch_size}")

            edge_and_query_list = generate_examples(
                lookahead_size=lookahead,
                num_examples=sample_size,
                max_branches=branch_size,
                dedupe_graphs=dedupe_graphs,
                enforce_branch_size=enforce_branch_size,
                random_seed=args.seed,
                verbose=verbose
            )

            data_points = extract_data_points(
                edge_and_query_list,
                lookahead_size=lookahead,
                max_branches=branch_size
            )
            all_data_points.extend(data_points)

    # Generate output filename
    if args.output:
        output_filename = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = (
            f"{timestamp}_graph_data_"
            f"l_{lookaheads[0]}_to_{lookaheads[-1]}_"
            f"b_{branches[0]}_to_{branches[-1]}_"
            f"n_{sample_size}.json"
        )

    # Save data
    with open(output_filename, 'w') as f:
        json.dump(all_data_points, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Successfully generated {len(all_data_points)} samples")
        print(f"✓ Saved to: {output_filename}")
        print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
