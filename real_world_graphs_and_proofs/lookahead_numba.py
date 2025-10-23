"""
Optimized Numba-accelerated script for calculating lookahead acorss all
Source-target pairs.
"""
import os, time, argparse, psutil, sys
from collections import defaultdict
import numpy as np, numba
from numba import jit, prange
from tqdm import tqdm

@jit(nopython=True, cache=True)
def lookahead_single_mask(edges_to, edge_idx, source, num_nodes):
    """
    Function for sources with <=63 children which only requires 1 int64 bitmask.
    Uses bitmask to track which primary children have been encountered for performance.
    This function calculates the lookahead between the sources and all other nodes.
    """

    # Initialize lookahead array with -1 (unreachable)
    lookahead = np.full(num_nodes, -1, dtype=np.int32)
    lookahead[source] = 0
    
    s0, s1 = edge_idx[source], edge_idx[source+1]
    k = s1 - s0  # number of source's children
    if k == 0:
        return lookahead
    
    # Start without dedup for performance
    use_dedup = False
    # Max limit of the size of the queue
    max_queue_size = min(num_nodes * 3, len(edges_to))
    queue = np.empty(max_queue_size, dtype=np.int32)
    
    # Pre-allocate dedup structures if needed
    in_next_queue = np.zeros(num_nodes, dtype=np.bool_)
    to_reset = np.empty(num_nodes, dtype=np.int32)

    # visited[i] = True if node i has been processed
    visited = np.zeros(num_nodes, dtype=np.bool_)
    # mask[i] = bitmask showing which source children reach node i
    mask = np.zeros(num_nodes, dtype=np.int64)
    
    # Mark source as visited
    visited[source] = True
    
    # Initialize queue with source's children
    queue_size = 0
    for i, e in enumerate(range(s0, s1)):
        c = edges_to[e]
        if queue_size < max_queue_size:
            queue[queue_size] = c
            queue_size += 1
        # Each child gets its own bit in the mask
        mask[c] = np.int64(1) << i
    
    current_lookahead = 1
    
    while queue_size > 0:
        # EARLY DETECTION: Switch if queue is getting large
        if not use_dedup and queue_size > num_nodes // 2:
            # print("Switching to dedup mode - large queue detected")
            use_dedup = True
        
        # Check if any source child appears in ALL nodes' masks in current queue
        nonempty_intersection = False
        
        for child_bit in range(k):
            bit_mask = np.int64(1) << child_bit
            all_have_bit = True
            for i in range(queue_size):
                v = queue[i]
                if not (mask[v] & bit_mask):
                    all_have_bit = False
                    break
            
            if all_have_bit:
                nonempty_intersection = True
                break
        
        if nonempty_intersection:
            # All remaining nodes will have the same lookahead value
            break
        
        # Process current level
        if use_dedup:
            new_queue = np.empty(num_nodes, dtype=np.int32)
        else:
            new_queue = np.empty(max_queue_size, dtype=np.int32)
            
        new_queue_size = 0
        to_reset_count = 0
        
        # Clear in_next_queue if using dedup
        if use_dedup:
            # Reset from previous iteration
            for i in range(queue_size):
                in_next_queue[queue[i]] = False
        
        for i in range(queue_size):
            vertex = queue[i]
            if visited[vertex]:
                continue
                
            visited[vertex] = True
            lookahead[vertex] = current_lookahead
            vertex_mask = mask[vertex]
            
            # Enqueue children
            v0, v1 = edge_idx[vertex], edge_idx[vertex+1]
            for e in range(v0, v1):
                c = edges_to[e]
                
                if use_dedup:
                    # Only add to queue if not already there
                    if not in_next_queue[c]:
                        in_next_queue[c] = True
                        new_queue[new_queue_size] = c
                        new_queue_size += 1
                else:
                    if new_queue_size < max_queue_size:
                        new_queue[new_queue_size] = c
                        new_queue_size += 1
                
                # Always merge masks
                mask[c] |= vertex_mask
        
        # Update queue
        queue[:new_queue_size] = new_queue[:new_queue_size]
        queue_size = new_queue_size
        current_lookahead += 1
    
    # Process remaining nodes in queue with same lookahead
    for i in range(queue_size):
        vertex = queue[i]
        if not visited[vertex]:
            visited[vertex] = True
            lookahead[vertex] = current_lookahead
            
            # BFS for remaining nodes
            bfs_queue = np.empty(num_nodes, dtype=np.int32)
            bfs_queue[0] = vertex
            bfs_head = 0
            bfs_tail = 1
            
            while bfs_head < bfs_tail:
                v = bfs_queue[bfs_head]
                bfs_head += 1
                
                v0, v1 = edge_idx[v], edge_idx[v+1]
                for e in range(v0, v1):
                    c = edges_to[e]
                    if not visited[c]:
                        visited[c] = True
                        lookahead[c] = current_lookahead
                        if bfs_tail < num_nodes:
                            bfs_queue[bfs_tail] = c
                            bfs_tail += 1
    
    return lookahead

@jit(nopython=True, cache=True)
def lookahead_multi_mask(edges_to, edge_idx, source, num_nodes):
    """
    Function for sources with >63 children which requires multiple int64 bitmask.
    Uses bitmask to track which primary children have been encountered for performance.
    This function calculates the lookahead between the sources and all other nodes.
    """

    # Initialize lookahead array with -1 (unreachable)
    lookahead = np.full(num_nodes, -1, dtype=np.int32)
    lookahead[source] = 0
    
    s0, s1 = edge_idx[source], edge_idx[source+1]
    k = s1 - s0  # number of source's children
    if k == 0:
        return lookahead
    
    # Calculate number of masks needed
    m = (k + 62) // 63
    
    # For very high degree nodes, start with dedup immediately
    use_dedup = k > 1000
    
    if use_dedup:
        max_queue_size = num_nodes
        queue = np.empty(max_queue_size, dtype=np.int32)
    else:
        max_queue_size = min(num_nodes * 5, len(edges_to))
        queue = np.empty(max_queue_size, dtype=np.int32)
    
    # Pre-allocate dedup structures
    in_next_queue = np.zeros(num_nodes, dtype=np.bool_)
    to_reset = np.empty(num_nodes, dtype=np.int32)

    # visited[i] = True if node i has been processed
    visited = np.zeros(num_nodes, dtype=np.bool_)
    # node_mask: flattened array of bitmasks
    # For node i: segments are at indices [i*m, i*m+1, ..., i*m+m-1]
    node_mask = np.zeros(num_nodes * m, dtype=np.int64)
    
    # Mark source as visited
    visited[source] = True
    
    # Initialize queue with source's children
    queue_size = 0
    for i, e in enumerate(range(s0, s1)):
        c = edges_to[e]
        
        if use_dedup:
            if not in_next_queue[c]:
                # Only add if not already in queue
                in_next_queue[c] = True
                queue[queue_size] = c
                queue_size += 1
        else:
            if queue_size < max_queue_size:
                queue[queue_size] = c
                queue_size += 1
                
        # Set bit in appropriate mask segment
        seg_idx, bit_pos = divmod(i, 63)
        node_mask[c * m + seg_idx] = np.int64(1) << bit_pos
    
    current_lookahead = 1
    
    while queue_size > 0:
        # EARLY DETECTION: Switch if queue is getting large
        if not use_dedup and queue_size > num_nodes // 2:
            # print("Multi: Switching to dedup mode - large queue detected")
            use_dedup = True
            
            # Need to deduplicate current queue before switching
            new_queue_temp = np.empty(num_nodes, dtype=np.int32)
            new_size = 0
            
            # Mark and deduplicate current queue
            for i in range(queue_size):
                node = queue[i]
                if not in_next_queue[node]:
                    in_next_queue[node] = True
                    new_queue_temp[new_size] = node
                    new_size += 1
            
            # Clear marks for next iteration
            for i in range(new_size):
                in_next_queue[new_queue_temp[i]] = False
            
            queue = new_queue_temp
            queue_size = new_size
            max_queue_size = num_nodes

        # Check if any source child appears in ALL nodes' masks in current queue
        nonempty_intersection = False
        
        for child_idx in range(k):
            seg_idx, bit_pos = divmod(child_idx, 63)
            bit_mask = np.int64(1) << bit_pos
            
            all_have_bit = True
            for i in range(queue_size):
                v = queue[i]
                if not (node_mask[v * m + seg_idx] & bit_mask):
                    all_have_bit = False
                    break
            
            if all_have_bit:
                nonempty_intersection = True
                break
        
        if nonempty_intersection:
            break
        
        # Process current level
        if use_dedup:
            new_queue = np.empty(num_nodes, dtype=np.int32)
        else:
            new_queue = np.empty(max_queue_size, dtype=np.int32)
            
        new_queue_size = 0
        to_reset_count = 0
        
        for i in range(queue_size):
            vertex = queue[i]
            if visited[vertex]:
                continue
                
            visited[vertex] = True
            lookahead[vertex] = current_lookahead
            v_base = vertex * m
            
            # Enqueue children
            v0, v1 = edge_idx[vertex], edge_idx[vertex+1]
            for e in range(v0, v1):
                c = edges_to[e]
                
                if use_dedup:
                    if not in_next_queue[c]:
                        in_next_queue[c] = True
                        new_queue[new_queue_size] = c
                        new_queue_size += 1
                        to_reset[to_reset_count] = c
                        to_reset_count += 1
                else:
                    if new_queue_size < max_queue_size:
                        new_queue[new_queue_size] = c
                        new_queue_size += 1
                
                # Always merge masks
                c_base = c * m
                for seg in range(m):
                    node_mask[c_base + seg] |= node_mask[v_base + seg]
        
        if use_dedup:
            # Reset only the nodes we marked
            for i in range(to_reset_count):
                in_next_queue[to_reset[i]] = False
        
        # Update queue
        queue[:new_queue_size] = new_queue[:new_queue_size]
        queue_size = new_queue_size
        current_lookahead += 1
    
    # Process remaining nodes in queue with same lookahead
    for i in range(queue_size):
        vertex = queue[i]
        if not visited[vertex]:
            visited[vertex] = True
            lookahead[vertex] = current_lookahead
            
            # BFS for remaining nodes
            bfs_queue = np.empty(num_nodes, dtype=np.int32)
            bfs_queue[0] = vertex
            bfs_head = 0
            bfs_tail = 1
            
            while bfs_head < bfs_tail:
                v = bfs_queue[bfs_head]
                bfs_head += 1
                
                v0, v1 = edge_idx[v], edge_idx[v+1]
                for e in range(v0, v1):
                    c = edges_to[e]
                    if not visited[c]:
                        visited[c] = True
                        lookahead[c] = current_lookahead
                        if bfs_tail < num_nodes:
                            bfs_queue[bfs_tail] = c
                            bfs_tail += 1
    
    return lookahead

# Select whether to use the single mask (<=63 outdegree) or multi mask (>63 outdegree) version of the algo
@jit(nopython=True, cache=True)
def lookahead_algo_selector(edges_to, edge_idx, source, num_nodes):
    if (edge_idx[source+1] - edge_idx[source]) <= 63:
        return lookahead_single_mask(edges_to, edge_idx, source, num_nodes)
    else:
        return lookahead_multi_mask(edges_to, edge_idx, source, num_nodes)

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def parallel_lookahead(edges_to, edge_idx, sources, num_nodes, max_val=1000):
    """
    Parallel computation of lookahead values for multiple source nodes.
    """
    n_src = len(sources)
    n_threads = numba.config.NUMBA_NUM_THREADS

    # Thread-local storage arrays to avoid race conditions
    # Thread lookahead distribution
    thread_counts = np.zeros((n_threads, max_val+1), dtype=np.int64)
    # Total pairs processed by threads
    thread_total = np.zeros(n_threads, dtype=np.int64)
    # Number of multi-mask sources (oudegree > 63)
    thread_multi_mask  = np.zeros(n_threads, dtype=np.int32)

    for s in prange(n_src):
        tid = s % n_threads
        src = sources[s]

        # Calculate out-degree
        deg = edge_idx[src+1] - edge_idx[src]

        # Debug print for each source
        if n_threads == 1:  # Only print in single-threaded mode
            print("Processing source", src, "with degree", deg)

        if deg > 63:
            thread_multi_mask[tid] += 1

        lookahead = lookahead_algo_selector(edges_to, edge_idx, src, num_nodes)

        if n_threads == 1:
            print("Completed source", src)

        # Collect statistics for all target nodes
        for t in range(num_nodes):
            if t == src: continue
            v = lookahead[t]
            if 0 <= v <= max_val:
                thread_counts[tid, v] += 1
                thread_total[tid]    += 1

    # Aggregate results from all threads
    dist = np.zeros(max_val+1, dtype=np.int64)
    pairs = 0
    mm = 0
    for tid in range(n_threads):
        dist  += thread_counts[tid]
        pairs += thread_total[tid]
        mm    += thread_multi_mask[tid]
    return dist, pairs, mm

@jit(nopython=True, cache=True)
def create_csr_arrays_fast(edge_list, num_nodes):
    """
    Fast Compressed Sparse Row array (fast graph representation) using Numba.
    """
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


def load_graph_optimized(input_path, max_lines=None):
    """
    Load graph and convert to CSR format in two passes through file
    First to count nodes and edges for allocating arrays,
    and second to build the edge list.
    """
    print(f"Loading graph from {input_path}", flush=True)

    # First pass: count nodes and edges
    node_set = set()
    edge_count = 0

    with open(input_path, 'rt', encoding='utf-8') as f:
        lines_processed = 0
        for line in tqdm(f, desc="Counting"):
            if line.startswith('#') or not line.strip():
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

    # Second pass: build edge list
    edge_arr = np.empty((edge_count, 2), dtype=np.int32)

    e = 0
    with open(input_path, 'rt', encoding='utf-8') as f:
        lines_processed = 0
        for line in tqdm(f, desc="Building"):
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            edge_arr[e, 0] = node_to_idx[parts[0]]
            edge_arr[e, 1] = node_to_idx[parts[2]]
            e += 1

            lines_processed += 1
            if max_lines and lines_processed >= max_lines:
                break

    edge_arr = edge_arr[:e]

    edges_to, edge_idx = create_csr_arrays_fast(edge_arr, num_nodes)

    # degree report
    deg = edge_idx[1:] - edge_idx[:-1]
    print(f"Nodes: {num_nodes:,}  ▏ Edges: {e:,}")
    print(f"Max out-degree: {deg.max()}  "
          f"(≤63: {(deg <= 63).sum():,}  >63: {(deg > 63).sum():,})")

    return edges_to, edge_idx, num_nodes, node_to_idx

def calculate_lookaheads_optimized(edges_to, edge_idx, num_nodes,
                                   batch_size, seed=None, preselected_sources=None):
    """
    Optimized lookahead calculation using batches and threading with Numba.
    """
    if preselected_sources is not None:
        all_nodes = preselected_sources.astype(np.int32)
    else:
        all_nodes = np.arange(num_nodes, dtype=np.int32)

    # Set a limit to the maximum lookahead value (Change to increase value)
    max_val = min(20000, num_nodes)
    lookahead_dist = np.zeros(max_val+1, dtype=np.int64)

    num_batches = (len(all_nodes) + batch_size - 1) // batch_size
    print(f"Processing {len(all_nodes):,} nodes in {num_batches} batches", flush=True)
    print(f"Using {numba.config.NUMBA_NUM_THREADS} threads", flush=True)

    tot_pairs = mm_sources = 0
    t0 = time.perf_counter()
    for batch_idx in range(num_batches):
        # Get the sources for this batch
        sl  = slice(batch_idx*batch_size, (batch_idx+1)*batch_size)
        batch_nodes  = all_nodes[sl]

        d,p,mm = parallel_lookahead(edges_to, edge_idx, batch_nodes,
                                    num_nodes, max_val)

        lookahead_dist += d
        tot_pairs += p
        mm_sources += mm

        done = (batch_idx+1)*batch_size
        rate = done/(time.perf_counter()-t0)
        print(f"Batch {batch_idx+1}/{num_batches} ▏ {done:,}/{len(all_nodes):,} "
              f"▏ {rate:,.0f} src/s ▏ multi-mask {mm_sources}", flush=True)

    avg = (lookahead_dist*np.arange(max_val+1)).sum()/tot_pairs if tot_pairs else -1
    return lookahead_dist, avg, tot_pairs

# Save results to a file
def save_results(dist, avg, total, path):
    with open(path, 'w') as f:
        f.write("# Look-ahead distribution\nValue\tCount\tPercent\n")
        for v,c in dist.items():
            f.write(f"{v}\t{c}\t{c/total*100:.6f}\n")
    print(f"Results written to {path}")

# Test on various sammple graphs to make sure it's the same results as python verison
from textwrap import dedent
def run_test_suite():
    print("Running test samples…")
    print("="*70)
    test_graphs = {
        'Graph 1 (Diamond)': {
            'edges': [(0,1), (0,2), (0,3), (1,4), (2,4), (4,5), (3,5)],
            'nodes': ['S', 'A', 'B', 'D', 'C', 'E'],
            'expected': [0, 1, 1, 1, 2, 2]
        },
        'Graph 2 (Complex)': {
            'edges': [(0,1), (0,2), (0,3), (1,4), (2,4), (4,7), (4,6),
                      (3,5), (6,8), (8,5), (5,9)],
            'nodes': ['S', 'A', 'B', 'D', 'C', 'E', 'F', 'H', 'G', 'I'],
            'expected': [0, 1, 1, 1, 2, 2, 3, 3, 4, 3]
        },
        'Graph Star': {
            'edges': [(0,1), (0,2), (0,3), (0,4), (1,5), (5,6), (6,7)],
            'nodes': ['S', 'A', 'B', 'D', 'E', 'F', 'G', 'H'],
            'expected': [0, 1, 1, 1, 1, 2, 2, 2]
        },
        'Graph Chain': {
            'edges': [(0,1), (1,2), (2,3), (3,4), (4,5)],
            'nodes': ['S', 'A', 'B', 'C', 'D', 'E'],
            'expected': [0, 1, 1, 1, 1, 1]
        },
        'Graph Forks': {
            'edges': [(0,1), (1,2), (1,3), (3,4), (3,5)],
            'nodes': ['S', 'A', 'B', 'C', 'D', 'E'],
            'expected': [0, 1, 1, 1, 1, 1]
        },
        'Graph Cycle4': {
            'edges': [(0,1), (1,2), (2,0), (0,3), (3,0)],
            'nodes': ['S', 'A', 'B', 'C'],
            'expected': [0, 1, 2, 1]
        },
        'Graph Bug Example': {
            'edges': [(0,1),(0,2),(0,3),(1,4),(2,4),(2,5),
                      (3,5),(4,6),(5,6),(6,7)],
            'nodes': ['S','A','B','C','D','E','F','G'],
            'expected': [0,1,1,1,2,2,2,2]
        },
        'Graph Superset Path': {
            'edges': [(0,1), (0,2), (1,3), (2,3), (1,4), (3,4)],
            'nodes': ['S', 'A', 'B', 'C', 'D'],
            'expected': [0, 1, 1, 2, 2]
        },
        'Graph Triple Convergence': {
            'edges': [(0,1), (0,2), (0,3), (1,4), (2,4), (3,4), 
                      (1,5), (2,5), (3,5), (4,6), (5,6)],
            'nodes': ['S', 'A', 'B', 'C', 'D', 'E', 'F'],
            'expected': [0, 1, 1, 1, 2, 2, 2]
        },
        'Graph Order A First': {
            # Vertex order: S(0) A(1) B(2) C(3) D(4) E(5)
            'edges': [(0,1), (0,2),
                    (1,3), (2,3),
                    (2,4),
                    (3,5), (4,5)],
            'nodes': ['S', 'A', 'B', 'C', 'D', 'E'],
            'expected': [0, 1, 1, 2, 2, 2]   # S A B C D E
        },
        'Graph Order B First': {
            # Vertex order: S(0) A(1) B(2) C(3) D(4) E(5)
            'edges': [(0,2), (0,1),
                    (2,3), (2,4),
                    (1,3),
                    (3,5), (4,5)],
            'nodes': ['S', 'A', 'B', 'C', 'D', 'E'],
            'expected': [0, 1, 1, 2, 2, 2]   # S A B C D E
        },
        'Diamond': {
            'edges': [(0,1), (0,2), (2,3), (1,3)],
            'nodes': ['S', 'A', 'B', 'C'],
            'expected': [0, 1, 1, 2]
        }
    }

    all_ok = True
    for name, data in test_graphs.items():
        print(f"\n{name}:")
        n_nodes = len(data['nodes'])
        arr = np.array(data['edges'], dtype=np.int32)
        e_to, e_idx = create_csr_arrays_fast(arr, n_nodes)
        res = lookahead_algo_selector(e_to, e_idx, 0, n_nodes)
        ok = True
        for i, exp in enumerate(data['expected']):
            if res[i] != exp:
                print(f"INCORRECT: {data['nodes'][i]}  exp {exp}  got {res[i]}")
                ok = all_ok = False
        if ok:
            print(f"CORRECT {list(res[:n_nodes])}")

    print("\nMulti-mask (>63 children) check:")
    big_arr = np.array([(0,i) for i in range(1,71)], dtype=np.int32)
    e_to, e_idx = create_csr_arrays_fast(big_arr, 71)
    r = lookahead_algo_selector(e_to, e_idx, 0, 71)
    if r[0]==0 and np.all(r[1:]==1):
        print("multi-mask branch OK")
    else:
        print(" multi-mask failed"); all_ok=False

    print("\n" + "="*70)
    print("ALL TESTS PASSED" if all_ok else "Some tests failed")
    return all_ok


def main():
    ap = argparse.ArgumentParser(
        description="Lookahead calculation across all pairs (fast Numba version)")
    ap.add_argument('--input',  type=str, help='Triplet TSV file')
    ap.add_argument('--output', type=str, default='lookahead_results.txt')
    ap.add_argument('--batch-size', type=int, default=50_000)
    ap.add_argument('--max-lines', type=int)
    ap.add_argument('--threads', type=int)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--sample-count', type=int,
                help='Exact number of random source nodes to process (sampling)')
    args = ap.parse_args()

    if args.test:
        run_test_suite()
        return

    if args.threads:
        os.environ['NUMBA_NUM_THREADS'] = str(args.threads)
        numba.config.NUMBA_NUM_THREADS = args.threads

    edges_to, edge_idx, num_nodes, _ = load_graph_optimized(
        args.input, args.max_lines)

    if args.sample_count is not None:
        rng = np.random.default_rng(args.seed)
        sample_count = min(args.sample_count, num_nodes)
        src_nodes = rng.choice(np.arange(num_nodes, dtype=np.int32),
                            size=sample_count, replace=False)
        print(f"Sampling exactly {sample_count:,} random source nodes "
            f"({sample_count/num_nodes*100:.4f} %)", flush=True)
    else:
        src_nodes = None

    dist, avg, total = calculate_lookaheads_optimized(
        edges_to, edge_idx, num_nodes,
        batch_size=args.batch_size,
        seed=args.seed,
        preselected_sources=src_nodes
    )

    dist_dict = {v:int(c) for v,c in enumerate(dist) if c}
    save_results(dist_dict, avg, total, args.output)

    print(f"\nTotal pairs processed: {total:,}")
    print(f"Average value: {avg:.3f}")

if __name__ == '__main__':
    main()