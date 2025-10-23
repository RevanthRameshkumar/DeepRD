#!/usr/bin/env python3
"""
Generate lookahead and branch histograms for real-world graphs and proofs.

This script creates a two-row histogram visualization comparing lookahead
and branching distributions across various real-world graph datasets.
"""

import os
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_proof_lookahead_data(output_file='proof_lookahead_results_full.tsv'):
    """
    Generate proof lookahead data from NaturalProofs JSON files.

    Loads theorems from three NaturalProofs datasets and counts the distribution
    of proof lengths (number of content lines per proof).

    Args:
        output_file: Path to save the TSV file

    Returns:
        Path to the generated TSV file
    """
    if os.path.exists(output_file):
        print(f"✓ {output_file} already exists, skipping generation")
        return output_file

    print(f"Generating {output_file} from NaturalProofs datasets...")

    # Load all three NaturalProofs datasets
    json_files = [
        'data/naturalproofs_proofwiki.json',
        'data/naturalproofs_stacks.json',
        'data/naturalproofs_trench.json'
    ]

    dataset = []
    for json_file in json_files:
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Required file not found: {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            dataset.extend(data['dataset']['theorems'])

    print(f"  Loaded {len(dataset)} theorems")

    # Count proof lengths
    total = Counter()
    for theorem in dataset:
        for proof in theorem['proofs']:
            proof_length = len(proof['contents'])
            if proof_length > 0:
                total.update([proof_length])

    # Convert to DataFrame
    sorted_items = sorted(total.items())
    df = pd.DataFrame(sorted_items, columns=['Value', 'Count'])
    total_count = df['Count'].sum()
    df['percent_new'] = df['Count'] / total_count * 100
    df['Percent'] = df['percent_new']

    # Save to TSV
    df.to_csv(output_file, sep='\t', index=False)
    print(f"✓ Generated {output_file} with {len(df)} unique proof lengths")
    print(f"  Total proofs: {total_count}")
    print(f"  Mean proof length: {(df['Value'] * df['Count']).sum() / total_count:.2f}")

    return output_file


def _load_normalized(path, value_col, count_col):
    """
    Helper: load either a single file or all files in a directory,
    normalize each file to a probability distribution, and sum.
    Returns a DataFrame with [value_col, count] probabilities.
    """
    def _read_and_norm(fp):
        df = pd.read_csv(fp, sep=r'\s+', comment='#', usecols=[value_col, count_col])
        total = df[count_col].sum()
        df[count_col] = df[count_col] / total
        return df

    if os.path.isdir(path):
        dfs = []
        for fn in os.listdir(path):
            fp = os.path.join(path, fn)
            if os.path.isfile(fp):
                try:
                    dfs.append(_read_and_norm(fp))
                except Exception:
                    pass
        if not dfs:
            raise ValueError(f"No valid files in {path!r}")
        df = pd.concat(dfs, ignore_index=True)
        df = df.groupby(value_col, as_index=False)[count_col].mean()
    else:
        df = pd.read_csv(path, sep=r'\s+', comment='#', usecols=[value_col, count_col])
        df[count_col] = df[count_col] / df[count_col].sum()
    return df


def plot_lookahead_and_branch_histograms(
    lookahead_sources,
    branch_sources,
    quantiles=None,
    figsize=(4, 8),
    output_file='lookahead_branch_histograms.pdf'
):
    """
    Plot two rows of histograms in a shared figure:
      - Row 1: lookahead histograms (len(lookahead_sources) panels, centered)
      - Row 2: branch histograms (len(branch_sources) panels, centered)

    Each source may be a file or a directory. Y-axis is percent.
    quantiles: list of floats in [0, 1], e.g. [0.9, 0.95], to draw vertical lines
    """
    if quantiles is None:
        quantiles = []

    n_look = len(lookahead_sources)
    n_branch = len(branch_sources)
    m = max(n_look, n_branch)

    fig, axes = plt.subplots(2, m, figsize=(figsize[0]*m, figsize[1]), sharey='row')
    axes = axes if m > 1 else axes.reshape(2, 1)

    # Initialize all axes to invisible
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    def _plot_row(sources, row_idx, value_col, count_col, x_label):
        offset = (m - len(sources)) // 2
        for i, (src, title) in enumerate(sources):
            df = _load_normalized(src, value_col=value_col, count_col=count_col)
            df = df.sort_values(by=value_col)
            df['percent'] = df[count_col] * 100

            # compute cumulative distribution for quantile lookup
            probs = df[count_col].to_numpy()
            values = df[value_col].to_numpy()
            cum_probs = np.cumsum(probs)

            ax = axes[row_idx, offset + i]
            ax.set_visible(True)
            ax.bar(df[value_col], df['percent'], width=1.0, align='center')
            ax.set_xlabel(value_col, fontsize=16)
            ax.set_yscale('log')
            ax.set_xscale('log')
            if offset + i == 0:
                ax.set_ylabel('percent', fontsize=16)
            ax.set_xlabel(x_label, fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.set_title(title, fontsize=16)

            # draw vertical lines at each requested quantile
            for q in quantiles:
                if 0 <= q <= 1:
                    x_q = np.interp(q, cum_probs, values)
                    ax.axvline(x_q, linestyle='--', color='red')

    # Plot lookahead row
    _plot_row(lookahead_sources, row_idx=0, value_col='Value', count_col='Count', x_label='Lookahead')
    # Plot branch row
    _plot_row(branch_sources, row_idx=1, value_col='children_count', count_col='count', x_label='Number of Branches')

    plt.tight_layout()
    fig.savefig(output_file, format='pdf', bbox_inches='tight', dpi=600)
    print(f"✓ Saved figure to {output_file}")
    plt.show()


def main():
    """Main entry point for generating histograms."""

    # Generate proof lookahead data if it doesn't exist
    generate_proof_lookahead_data()

    # Define data sources for lookahead analysis
    lookahead_sources = [
        ("data/conceptnet-5.7.0_lookahead_results_300k.txt", 'ConceptNet'),
        ("data/ogbl-wikikg2_lookahead_results.txt", 'ogbl-wikikg2'),
        ("data/ogbn-papers100M_lookahead_results_30k.txt", 'ogbn-papers100M'),
        ("data/agg_lookahead", 'ogb'),  # directory
        ("proof_lookahead_results_full.tsv", 'NaturalProofs')
    ]

    # Define data sources for branch analysis
    branch_sources = [
        ("data/branch_count_conceptnet-5.7.0.tsv", 'ConceptNet'),
        ("data/branch_count_ogbl-wikikg2.tsv", 'ogbl-wikikg2'),
        ("data/branch_count_ogbn-papers100M.tsv", 'ogbn-papers100M'),
        ("data/agg_branches", 'ogb')  # directory
    ]

    # Generate the histograms with quantile lines
    plot_lookahead_and_branch_histograms(
        lookahead_sources,
        branch_sources,
        quantiles=[0.5, 0.75, 0.9, 0.99, 0.999],
        output_file='lookahead_branch_histograms.pdf'
    )


if __name__ == '__main__':
    main()
