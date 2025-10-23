"""
Optimized Numba-accelerated script for getting pairwise distance distribution across all
Source-target pairs.
Key optimizations:
- Numba for JIT compilation and true multithreading
- Shared memory for the graph
- Use int32 for arrays to save memory
- Pre-allocated and reusable buffers
- Use CSR format to represent the graph efficiently
- Parallel BFS for distance rather than rustworkx

"""

import numpy as np
import numba
from numba import jit, prange, types
from numba.typed import Dict
import rustworkx as rx
import os
import time
from tqdm import tqdm
from collections import Counter
import argparse
import pickle
import psutil
from pathlib import Path
import shutil

# Optimized BFS that reuses pre-allocated arrays.
@jit(nopython=True, cache=True, fastmath=True)
def bfs_optimized(edges_to, edge_indices, source, num_nodes, 
                  distances, visited, queue):
    """
    Args:
        edges_to: CSR format edge destinations
        edge_indices: CSR format edge indices
        source: Source node
        num_nodes: Total nodes
        distances: Pre-allocated distance array (reused)
        visited: Pre-allocated visited array (reused)
        queue: Pre-allocated queue array (reused)
    
    Returns:
        Number of reachable nodes
    """
    # Reset arrays (faster than allocating new ones)
    distances[:] = -1
    visited[:] = False
    
    # Initialize
    distances[source] = 0
    visited[source] = True
    queue[0] = source
    queue_start = 0
    queue_end = 1
    reachable = 1
    
    while queue_start < queue_end:
        current = queue[queue_start]
        queue_start += 1
        current_dist = distances[current]
        
        # Get neighbors from sparse representation
        start_idx = edge_indices[current]
        end_idx = edge_indices[current + 1]
        
        for edge_idx in range(start_idx, end_idx):
            neighbor = edges_to[edge_idx]
            if not visited[neighbor]:
                visited[neighbor] = True
                distances[neighbor] = current_dist + 1
                queue[queue_end] = neighbor
                queue_end += 1
                reachable += 1
    
    return reachable

# Call BFS in parallel across multiple sources, and accumulate results after.
@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def parallel_bfs_optimized(edges_to, edge_indices, sources, num_nodes, max_dist=5000):
    """    
    Args:
        edges_to: CSR edge array
        edge_indices: CSR index array
        sources: Array of source nodes
        num_nodes: Total number of nodes
        max_dist: Maximum distance to track

    Returns:
        Distribution of distances
    """
    n_sources = len(sources)
    n_threads = numba.config.NUMBA_NUM_THREADS
    
    # Thread-local storage for distance counts
    # Each thread gets its own row
    thread_counts = np.zeros((n_threads, max_dist), dtype=np.int64)
    
    # Process sources in parallel
    for source_idx in prange(n_sources):
        thread_id = source_idx % n_threads
        src = sources[source_idx]
        
        # Pre-allocate arrays for this thread
        distances = np.full(num_nodes, -1, dtype=np.int32)
        visited = np.zeros(num_nodes, dtype=np.bool_)
        queue = np.zeros(num_nodes, dtype=np.int32)
        
        # Run optimized BFS
        reachable = bfs_optimized(edges_to, edge_indices, src, num_nodes,
                                  distances, visited, queue)
        
        # Count distances in thread-local storage
        for i in range(num_nodes):
            dist = distances[i]
            if 0 <= dist < max_dist:
                thread_counts[thread_id, dist] += 1
    
    # Aggregate thread-local results
    total_counts = np.zeros(max_dist, dtype=np.int64)
    for t in range(n_threads):
        for d in range(max_dist):
            total_counts[d] += thread_counts[t, d]
    
    return total_counts

# Fast Compressed Sparse Row array (fast graph representation) using Numba.
@jit(nopython=True, cache=True)
def create_csr_arrays_fast(edge_list, num_nodes):
    num_edges = len(edge_list)
    
    # Count out-degrees
    out_degree = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_edges):
        src = edge_list[i, 0]
        out_degree[src] += 1
    
    # Build edge indices (cumulative sum)
    edge_indices = np.zeros(num_nodes + 1, dtype=np.int32)
    cumu_sum = 0
    for i in range(num_nodes):
        edge_indices[i] = cumu_sum
        cumu_sum += out_degree[i]
    edge_indices[num_nodes] = cumu_sum
    
    # Fill edges_to array
    edges_to = np.zeros(cumu_sum, dtype=np.int32)
    edge_pos = np.zeros(num_nodes, dtype=np.int32)
    
    for i in range(num_edges):
        src = edge_list[i, 0]
        dst = edge_list[i, 1]
        pos = edge_indices[src] + edge_pos[src]
        edges_to[pos] = dst
        edge_pos[src] += 1
    
    return edges_to, edge_indices

# Load graph and convert to CSR format in two passes through file
# First to count nodes and edges for allocating arrays,
# and second to build the edge list.
def load_graph_optimized(input_path, max_lines=None):
    print(f"Loading graph from {input_path}", flush=True)
    
    # First pass: count nodes and edges
    node_set = set()
    edge_count = 0
    
    with open(input_path, 'rt', encoding='utf-8') as f:
        lines_processed = 0
        for line in tqdm(f, desc="First pass - counting"):
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            node_set.add(parts[0])
            node_set.add(parts[2])
            edge_count += 1
            
            lines_processed += 1
            if max_lines and lines_processed >= max_lines:
                break
    
    # Create node mapping
    node_to_idx = {node: idx for idx, node in enumerate(sorted(node_set))}
    num_nodes = len(node_to_idx)
    
    print(f"Counted {num_nodes} unique nodes, {edge_count} edges", flush=True)
    
    # Second pass: build edge list
    edge_list = np.zeros((edge_count, 2), dtype=np.int32)
    
    edge_idx = 0
    with open(input_path, 'rt', encoding='utf-8') as f:
        lines_processed = 0
        for line in tqdm(f, desc="Second pass - building edges"):
            if not line or line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            u = node_to_idx[parts[0]]
            v = node_to_idx[parts[2]]
            
            edge_list[edge_idx] = [u, v]
            edge_idx += 1
            
            lines_processed += 1
            if max_lines and lines_processed >= max_lines:
                break
    
    # Trim edge list if needed
    edge_list = edge_list[:edge_idx]
    
    print(f"Building CSR arrays for {num_nodes} nodes and {edge_idx} edges", flush=True)
    edges_to, edge_indices = create_csr_arrays_fast(edge_list, num_nodes)
    
    return edges_to, edge_indices, num_nodes, node_to_idx

# Optimized distannce calculation using batches and threading with Numba.
# Print out results.
# Added code to keep track of performance.
def calculate_distances_optimized(edges_to, edge_indices, num_nodes,
                                  batch_size, sample_fraction=1.0):
    """
    Args:
        edges_to: CSR edge array
        edge_indices: CSR index array  
        num_nodes: Total number of nodes
        batch_size: Nodes per batch
        sample_fraction: Fraction of nodes to sample
    """

    # Sample nodes if requested
    all_nodes = np.arange(num_nodes, dtype=np.int32)
    if sample_fraction < 1.0:
        num_samples = int(num_nodes * sample_fraction)
        all_nodes = np.random.choice(all_nodes, size=num_samples, replace=False)
        print(f"Sampling {num_samples:,} nodes out of {num_nodes:,}", flush=True)
    
    # Initialize
    max_dist = min(5000, num_nodes) # cap at 5000 distance
    total_distance_counts = np.zeros(max_dist, dtype=np.int64)
    
    # Process in batches
    num_batches = (len(all_nodes) + batch_size - 1) // batch_size
    print(f"Processing {len(all_nodes):,} nodes in {num_batches} batches", flush=True)
    print(f"Using {numba.config.NUMBA_NUM_THREADS} threads", flush=True)
    
    # Memory and performance tracking
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**3)
    start_time = time.perf_counter()
    
    # Progress bar
    pbar = tqdm(total=len(all_nodes), desc="Processing nodes")
    
    batch_times = []
    
    for batch_idx in range(num_batches):
        batch_start_time = time.perf_counter()
        
        # Get batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_nodes))
        batch_sources = all_nodes[start_idx:end_idx]
        
        # Run optimized parallel BFS
        batch_counts = parallel_bfs_optimized(
            edges_to, edge_indices, batch_sources, num_nodes, max_dist
        )
        
        total_distance_counts += batch_counts
        pbar.update(len(batch_sources))
        
        batch_time = time.perf_counter() - batch_start_time
        batch_times.append(batch_time)
        
        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            rate = end_idx / elapsed
            eta = (len(all_nodes) - end_idx) / rate if rate > 0 else 0
            mem_current = process.memory_info().rss / (1024**3)
            print(f"Progress: {end_idx:,}/{len(all_nodes):,} "
                  f"({rate:.1f} nodes/sec, ETA: {eta/60:.1f} min, "
                  f"Memory: {mem_current:.1f} GB)", flush=True)
    
    pbar.close()
    
    # Final statistics
    total_elapsed = time.perf_counter() - start_time
    mem_after = process.memory_info().rss / (1024**3)
    
    print(f"\nPerformance Summary:", flush=True)
    print(f"Total time: {total_elapsed:.2f} seconds", flush=True)
    print(f"Processing rate: {len(all_nodes)/total_elapsed:.1f} nodes/second", flush=True)
    print(f"Memory usage: {mem_before:.2f} GB -> {mem_after:.2f} GB", flush=True)
    
    # Convert to dictionary
    distribution = {}
    for dist, count in enumerate(total_distance_counts):
        if count > 0:
            distribution[dist] = int(count)
    
    return distribution

# Save distribution results as a text file.
def save_distribution_as_text(distribution, output_path):
    total_paths = sum(distribution.values())
    
    with open(output_path, 'w') as f:
        f.write("# Distance Distribution Results\n")
        f.write(f"# Total paths: {total_paths:,}\n")
        f.write("# Format: distance<tab>count<tab>percentage\n")
        f.write("#\n")
        f.write("Distance\tCount\tPercentage\n")
        
        for distance in sorted(distribution.keys()):
            count = distribution[distance]
            percentage = (count / total_paths) * 100
            f.write(f"{distance}\t{count}\t{percentage:.6f}\n")

# Debug function for calculating the distance for a single source
def analyze_single_source(edges_to, edge_indices, num_nodes, source_name, node_to_idx):
    # Find source index
    if source_name not in node_to_idx:
        print(f"Source node '{source_name}' not found in graph!", flush=True)
        return None
    
    source_idx = node_to_idx[source_name]
    print(f"Analyzing distances from '{source_name}' (index {source_idx})...", flush=True)
    
    # Pre-allocate arrays
    distances = np.full(num_nodes, -1, dtype=np.int32)
    visited = np.zeros(num_nodes, dtype=np.bool_)
    queue = np.zeros(num_nodes, dtype=np.int32)
    
    # Run BFS
    start_time = time.time()
    reachable = bfs_optimized(edges_to, edge_indices, source_idx, num_nodes,
                              distances, visited, queue)
    elapsed = time.time() - start_time
    
    print(f"BFS completed in {elapsed:.2f} seconds", flush=True)
    print(f"Reachable nodes: {reachable:,} out of {num_nodes:,}", flush=True)
    
    # Count distance distribution
    max_dist_found = 0
    distance_counts = {}
    
    for i in range(num_nodes):
        dist = distances[i]
        if dist >= 0:
            distance_counts[dist] = distance_counts.get(dist, 0) + 1
            if dist > max_dist_found:
                max_dist_found = dist
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Distance Analysis from '{source_name}'")
    print(f"{'='*60}")
    print(f"Total nodes: {num_nodes:,}")
    print(f"Reachable: {reachable:,} ({(reachable/num_nodes)*100:.2f}%)")
    print(f"Unreachable: {num_nodes-reachable:,} ({((num_nodes-reachable)/num_nodes)*100:.2f}%)")
    print(f"Maximum distance: {max_dist_found}")
    
    print(f"\nDistance Distribution:")
    print(f"{'Distance':>10} | {'Count':>12} | {'Percentage':>10}")
    print("-" * 40)
    
    cumulative_pct = 0
    for dist in sorted(distance_counts.keys()):
        count = distance_counts[dist]
        pct = (count / reachable) * 100
        cumulative_pct += pct
        
        print(f"{dist:10d} | {count:12,} | {pct:9.2f}%")
    
    return distance_counts, max_dist_found

def main():
    parser = argparse.ArgumentParser(description='Optimized Numba pairwise distance calculation')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to Wikidata triplet file')
    parser.add_argument('--output', type=str, default='pairwise_distances.txt',
                       help='Output file for results')
    parser.add_argument('--batch-size', type=int, required=True,
                       help='Nodes per batch')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                       help='Fraction of nodes to sample (0-1)')
    parser.add_argument('--max-lines', type=int, default=None,
                       help='Maximum lines to read from input')
    parser.add_argument('--threads', type=int, default=None,
                       help='Number of threads for Numba')
    parser.add_argument('--single-source', type=str, default=None,
                       help='Analyze distances from a single source node (like Q29387131)')
    
    args = parser.parse_args()
    
    # Set threads if specified
    if args.threads:
        os.environ['NUMBA_NUM_THREADS'] = str(args.threads)
        numba.config.NUMBA_NUM_THREADS = args.threads
    
    # Load and convert graph
    edges_to, edge_indices, num_nodes, node_to_idx = load_graph_optimized(
        args.input, args.max_lines
    )

    # If single source is specify only calculate the distnaces from that source
    if args.single_source:
        distance_counts, max_dist = analyze_single_source(
            edges_to, edge_indices, num_nodes, args.single_source, node_to_idx
        )
        return
    
    # Calculate pairwise distance
    distribution = calculate_distances_optimized(
        edges_to, edge_indices, num_nodes,
        batch_size=args.batch_size,
        sample_fraction=args.sample_fraction
    )
    
    # Save final results as a text file
    save_distribution_as_text(distribution, args.output)
    print(f"Results saved to {args.output}", flush=True)
    
    # Print summary
    print("\nDistance Distribution Summary:", flush=True)
    total_paths = sum(distribution.values())
    for distance in sorted(distribution.keys())[:20]:
        count = distribution[distance]
        percentage = (count / total_paths) * 100
        print(f"Distance {distance}: {count:,} paths ({percentage:.2f}%)", flush=True)

if __name__ == '__main__':
    print("Starting optimized Numba pair calculation...", flush=True)
    main()