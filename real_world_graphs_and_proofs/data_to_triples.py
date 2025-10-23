import os
import numpy as np
import rustworkx as rx
import torch

from ogb.nodeproppred import NodePropPredDataset
from ogb.graphproppred import GraphPropPredDataset
from ogb.linkproppred import LinkPropPredDataset
import urllib
import gzip

# 1. Where to store all OGB downloads
DATA_ROOT = os.path.join(os.getcwd(), "graph_data")
os.makedirs(DATA_ROOT, exist_ok=True)

# Directory for output triples
TRIPLES_DIR = os.path.join(DATA_ROOT, "data_triples")
os.makedirs(TRIPLES_DIR, exist_ok=True)

# How many graphs to sample per dataset when len(dataset) > 1 (set it to like 1000000000 if you just want everything for the multi graph datasets)
DATASET_COUNT = 1000000000

# Whether to prefix node IDs with graph ID for multi-graph datasets
# Set to True to avoid node ID collisions (e.g., "G0_1", "G1_1")
# Set to False to keep original node IDs (may have collisions)
USE_GRAPH_PREFIX = True

# Whether to include node type information for heterogeneous graphs like ogbl-biokg
# Set to True to output nodes as "T{type}:{id}" (e.g., "T0:12345" for type 0, node 12345)
# Set to False to output just node IDs
INCLUDE_NODE_TYPES = True

# Note: OGB dataset structure:
# - ogbn-* (node prediction): Usually 1 large graph, accessed as dataset[0]
#   - ogbn-mag is special: it's a heterogeneous graph with multiple node types (paper, author, 
#     institution, field_of_study) and edge types (writes, affiliated_with, has_topic, cites)
# - ogbg-* (graph prediction): Multiple graphs, accessed as dataset[idx] for idx in range(len(dataset))
# - ogbl-* (link prediction): Usually 1 graph with edge splits, special structure
#   - ogbl-biokg is special: it's a heterogeneous biomedical knowledge graph with typed nodes
#     Output format: T{type}:{node_id} for nodes (e.g., T0:12345 for a protein node)
#   - ogbl-wikikg2 is special: it's a large knowledge graph with relation types
#
# All graphs from each dataset are combined into a single .tsv file.
# For multi-graph datasets (ogbg-*), if USE_GRAPH_PREFIX is True, nodes are prefixed 
# with "G{graph_id}_" to avoid node ID collisions (e.g., "G0_1", "G1_1").
#
# Special datasets that might need attention:
# - ogbn-mag: Heterogeneous with edge_index_dict
# - ogbl-biokg: Uses head/tail/relation format instead of edge_index
# - ogbl-wikikg2: Large KG with many relation types
# - ogbl-vessel: 3D vessel tree, might have special properties

dataset_names = [
    "ogbn-products", "ogbn-proteins", "ogbn-arxiv", "ogbn-papers100M", "ogbn-mag",
    "ogbg-molhiv", "ogbg-molpcba", "ogbg-ppa", "ogbg-code2",
    "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox",
    "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molesol",
    "ogbg-molfreesolv", "ogbg-mollipo",
    "ogbl-ppa", "ogbl-collab", "ogbl-ddi", "ogbl-citation2",
    "ogbl-wikikg2", 
    "ogbl-biokg", 
    "ogbl-vessel"
]

rng = np.random.default_rng(seed=42)

for name in dataset_names:
    print(f"\n=== Processing {name} ===")
    
    # Check if output file for this dataset already exists
    fname = f"{name}.tsv"
    out_path = os.path.join(TRIPLES_DIR, fname)
    
    if os.path.exists(out_path):
        print(f"Skipping {name} - output file already exists: {fname}")
        continue

    # pick the right loader
    if name.startswith("ogbn-"):
        loader = NodePropPredDataset
    elif name.startswith("ogbg-"):
        loader = GraphPropPredDataset
    elif name.startswith("ogbl-"):
        loader = LinkPropPredDataset
    else:
        raise ValueError(f"Unknown OGB prefix for {name}")

    # load dataset
    dataset = loader(name=name, root=DATA_ROOT)
    print(f" → {len(dataset)} graph(s)")

    # Prepare single output file for this dataset
    fname = f"{name}.tsv"
    out_path = os.path.join(TRIPLES_DIR, fname)
    
    # Sample indices
    idxs = rng.choice(len(dataset), size=min(DATASET_COUNT, len(dataset)), replace=False)
    
    # For single-graph datasets, we only need to process once
    if name.startswith(("ogbn-", "ogbl-")):
        idxs = [0]  # Just process the single graph once
    
    # Open file once and write all graphs to it
    total_triples = 0
    last_progress_milestone = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, idx in enumerate(idxs):
            # Extract graph data based on dataset type
            edge_reltype = None  # Initialize for knowledge graphs
            head_types = None  # Initialize for heterogeneous graphs
            tail_types = None  # Initialize for heterogeneous graphs
            try:
                if name.startswith("ogbn-"):
                    # Node property prediction datasets
                    graph, label = dataset[0]  # Usually single graph
                    
                    # Special handling for heterogeneous graphs like ogbn-mag
                    if name == "ogbn-mag":
                        # ogbn-mag is heterogeneous with multiple edge types
                        # It has edge_index_dict instead of edge_index
                        edge_list = []
                        if "edge_index_dict" in graph:
                            print(f"Processing heterogeneous graph with edge types: {list(graph['edge_index_dict'].keys())}")
                            for edge_type, edges in graph["edge_index_dict"].items():
                                src, dst = edges[0], edges[1]
                                # Convert edge type to a simple string for the relation
                                edge_type_str = str(edge_type).replace(" ", "_").replace(",", "_")
                                edge_list.append((src, dst, edge_type_str))
                        else:
                            # DEBUG
                            print(f"Warning: Unexpected structure for {name}")
                            print(f"Available keys: {list(graph.keys())}")
                            continue
                    else:
                        # Regular homogeneous graph
                        edge_index = graph["edge_index"]
                elif name.startswith("ogbg-"):
                    # Graph property prediction datasets
                    graph, label = dataset[idx]
                    edge_index = graph["edge_index"]
                elif name.startswith("ogbl-"):
                    # Link property prediction datasets
                    # These have different structure
                    split_edge = dataset.get_edge_split()
                    edge_reltype = None
                    edge_index = None
                    
                    # Debug: print available splits
                    print(f"Available splits: {list(split_edge.keys())}")
                    
                    # Special handling for ogbl-biokg
                    if name == "ogbl-biokg" and "train" in split_edge:
                        print(f"Processing heterogeneous knowledge graph {name}")
                        print(f"Train split keys: {list(split_edge['train'].keys())}")
                        
                        if all(key in split_edge["train"] for key in ["head", "tail", "relation"]):
                            print(f"Found biokg format with head/relation/tail structure")
                            heads = split_edge["train"]["head"]
                            tails = split_edge["train"]["tail"]
                            relations = split_edge["train"]["relation"]
                            
                            # Create edge_index format
                            if hasattr(heads, 'numpy'):
                                heads = heads.numpy()
                                tails = tails.numpy()
                                relations = relations.numpy()
                            heads = np.asarray(heads).flatten()
                            tails = np.asarray(tails).flatten()
                            relations = np.asarray(relations).flatten()
                            
                            edge_index = np.vstack([heads, tails])
                            edge_reltype = relations
                            
                            print(f"Created edge_index with shape: {edge_index.shape}")
                            
                            # Also capture node types if available
                            if "head_type" in split_edge["train"] and "tail_type" in split_edge["train"]:
                                head_types = split_edge["train"]["head_type"]
                                tail_types = split_edge["train"]["tail_type"]
                                
                                # Convert to numpy
                                if hasattr(head_types, 'numpy'):
                                    head_types = head_types.numpy()
                                    tail_types = tail_types.numpy()
                                head_types = np.asarray(head_types).flatten()
                                tail_types = np.asarray(tail_types).flatten()
                                
                                print(f"Found node type information")
                                print(f"{len(np.unique(relations))} unique relations, {len(np.unique(head_types))} head types, {len(np.unique(tail_types))} tail types")
                    
                    # Standard knowledge graphs with edge_reltype
                    elif name in ["ogbl-wikikg2"] and "train" in split_edge:
                        print(f"Processing knowledge graph {name}")
                        if "edge_reltype" in split_edge["train"] and "edge" in split_edge["train"]:
                            edge_index = split_edge["train"]["edge"].T
                            edge_reltype = split_edge["train"]["edge_reltype"]
                            print(f"Found standard KG format with {len(np.unique(edge_reltype))} unique edge relation types")
                        elif "edge" in split_edge["train"]:
                            edge_index = split_edge["train"]["edge"].T
                        elif "edge_index" in split_edge["train"]:
                            edge_index = split_edge["train"]["edge_index"]
                    
                    # Regular link prediction datasets
                    else:
                        if "train" in split_edge:
                            if "edge" in split_edge["train"]:
                                edge_index = split_edge["train"]["edge"].T
                            elif "edge_index" in split_edge["train"]:
                                edge_index = split_edge["train"]["edge_index"]
                    
                    # Check if we successfully got edge data
                    if edge_index is None:
                        # DEBUG
                        print(f"Warning: Couldn't find edge data for {name}")
                        print(f"Available keys in train split: {list(split_edge['train'].keys()) if 'train' in split_edge else 'No train split'}")
                        continue
                else:
                    print(f"Warning: Unknown dataset type for {name}")
                    continue
                    
                # Handle heterogeneous graphs (ogbn-mag)
                if name == "ogbn-mag" and 'edge_list' in locals():
                    for src, dst, edge_type in edge_list:
                        # Convert to numpy if needed
                        if hasattr(src, 'numpy'):
                            src = src.numpy()
                            dst = dst.numpy()
                        src = np.asarray(src)
                        dst = np.asarray(dst)
                        
                        # Write triples with specific edge types
                        for u, v in zip(src.tolist() if hasattr(src, 'tolist') else src,
                                       dst.tolist() if hasattr(dst, 'tolist') else dst):
                            f.write(f"{u}\t{edge_type}\t{v}\n")
                        total_triples += len(src)
                    continue
                
                src, dst = edge_index[0], edge_index[1]
                
                # Convert to list (handles both numpy arrays and torch tensors)
                if hasattr(src, 'numpy'):
                    #torch tensor
                    src = src.numpy()
                    dst = dst.numpy()
                
                src = np.asarray(src)
                dst = np.asarray(dst)
                
                # Print total edges for large graphs
                if i == 0 and len(src) > 100000:
                    print(f"   • Processing {len(src):,} edges...")
                
                # Write triples for this graph
                for u, v in zip(src.tolist() if hasattr(src, 'tolist') else src, 
                               dst.tolist() if hasattr(dst, 'tolist') else dst):
                    if USE_GRAPH_PREFIX and len(idxs) > 1:
                        # multi-graph datasets prefix nodes with graph ID to avoid collisions
                        f.write(f"G{idx}_{u}\tP0\tG{idx}_{v}\n")
                    else:
                        # No prefix or single graph
                        f.write(f"{u}\tP0\t{v}\n")
                
                total_triples += len(src)
                
                # Progress indicator for multi-graph datasets
                if len(idxs) > 1 and (i + 1) % 100 == 0:
                    print(f"   • Processed {i + 1}/{len(idxs)} graphs...")
                
                # Progress indicator for large single graphs (every 1M edges)
                if total_triples >= last_progress_milestone + 1000000:
                    print(f"   • Processed {total_triples:,} triples...")
                    last_progress_milestone = total_triples
                    
            except Exception as e:
                # DEBUG
                print(f"Error processing graph {idx} in {name}: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue
    
    # Report total
    print(f"Wrote {total_triples:,} triples to {out_path} (from {len(idxs)} graph(s))")
    
    # Special output format notes
    if name == "ogbn-mag":
        print(f"Output format: node_id [tab] edge_type [tab] node_id")
    elif name == "ogbl-biokg" and INCLUDE_NODE_TYPES:
        print(f"Output format: T{{type}}:{{node_id}} [tab] R{{relation_id}} [tab] T{{type}}:{{node_id}}")
    elif name in ["ogbl-wikikg2"]:
        print(f"Output format: node_id [tab] R{{relation_id}} [tab] node_id")

# --- ConceptNet: download & write triples ---
CN_VERSION = "5.7.0"
CN_URL = (
    "https://s3.amazonaws.com/"
    "conceptnet/downloads/2019/edges/"
    f"conceptnet-assertions-{CN_VERSION}.csv.gz"
)
local_gz = os.path.join(DATA_ROOT, f"conceptnet-assertions-{CN_VERSION}.csv.gz")

# Check if ConceptNet output already exists
cn_out = os.path.join(TRIPLES_DIR, f"conceptnet-{CN_VERSION}.tsv")
if os.path.exists(cn_out):
    print(f"\nConceptNet triples already exist at {cn_out} - skipping processing")
else:
    # download if missing
    if not os.path.exists(local_gz):
        print(f"\nDownloading ConceptNet v{CN_VERSION} assertions…")
        urllib.request.urlretrieve(CN_URL, local_gz)
        print("Download complete.")
    
    print(f"Writing ConceptNet triples to {cn_out}…")
    
    with gzip.open(local_gz, "rt", encoding="utf-8") as gz, \
         open(cn_out, "w", encoding="utf-8") as out:
        for line in gz:
            parts = line.split('\t')
            if len(parts) >= 4:
                # URI of edge, relation, start, end, metadata…
                rel, start, end = parts[1], parts[2], parts[3]
                out.write(f"{start}\t{rel}\t{end}\n")
    
    print("ConceptNet triples written successfully.")