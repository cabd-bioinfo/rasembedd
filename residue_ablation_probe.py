#!/usr/bin/env python3
"""
Residue ablation probe script for protein cluster assignment interpretation.

Inputs:
- Protein sequences (TSV or FASTA)
- Protein language model (PLM) type (as in generate_embeddings.py)
- Cluster assignment file (TSV, as in linear_probe.py)
- Trained linear probe model (pickle, as in linear_probe.py)
- Wind    # Add sequence labels (smaller font size for protein names)
    seq_labels = [record.id for record in alignment]
    ax.set_yticks(range(num_seqs))
    label_fontsize = min(6, max(3, 60 / num_seqs))  # Smaller font for protein names
    ax.set_yticklabels(seq_labels, fontsize=label_fontsize)

    # Add position labels (reduce frequency for long alignments)
    tick_step = max(1, align_len // 15)  # At most 15 ticks
    tick_positions = range(0, align_len, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i+1) for i in tick_positions], fontsize=6)

    ax.set_title(f'{cluster_name} Cluster MSA - Colored by Normalized Ablation Score\n(Blue = Low, Red = High, Grey = Gaps)',
                fontsize=12, pad=10)r of amino acids to ablate per window)

Outputs:
- Per-residue ablation scores (TSV)
- Heatmap of ablation scores (PDF/PNG)
"""

import argparse
import math
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from io import StringIO

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio import AlignIO, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from matplotlib.backends.backend_pdf import PdfPages

# Import model loader from generate_embeddings.py
from generate_embeddings import load_model_class, load_sequences_from_fasta, load_sequences_from_tsv


# --- Helper functions ---
def load_linear_probe_model(model_path):
    with open(model_path, "rb") as f:
        probe = pickle.load(f)
    return probe


def mean_pool_embedding(embedding, mask=None):
    if mask is not None:
        embedding = embedding[mask]
    if embedding.shape[0] == 0:
        return np.zeros(embedding.shape[1])
    return embedding.mean(axis=0)


# --- Mutation policy helper ---
STANDARD_AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Minimal embedded BLOSUM62 scores for standard 20 AAs (symmetric). Values from common tables.
# For brevity and to avoid external dependency resolution issues, include a compact subset via dict-of-dicts.
# Note: This matrix is used only for relative sampling; exact values need not be exhaustive beyond 20x20.
BLOSUM62 = {
    "A": {
        "A": 4,
        "C": 0,
        "D": -2,
        "E": -1,
        "F": -2,
        "G": 0,
        "H": -2,
        "I": -1,
        "K": -1,
        "L": -1,
        "M": -1,
        "N": -2,
        "P": -1,
        "Q": -1,
        "R": -1,
        "S": 1,
        "T": 0,
        "V": 0,
        "W": -3,
        "Y": -2,
    },
    "C": {
        "A": 0,
        "C": 9,
        "D": -3,
        "E": -4,
        "F": -2,
        "G": -3,
        "H": -3,
        "I": -1,
        "K": -3,
        "L": -1,
        "M": -1,
        "N": -3,
        "P": -3,
        "Q": -3,
        "R": -3,
        "S": -1,
        "T": -1,
        "V": -1,
        "W": -2,
        "Y": -2,
    },
    "D": {
        "A": -2,
        "C": -3,
        "D": 6,
        "E": 2,
        "F": -3,
        "G": -1,
        "H": -1,
        "I": -3,
        "K": -1,
        "L": -4,
        "M": -3,
        "N": 1,
        "P": -1,
        "Q": 0,
        "R": -2,
        "S": 0,
        "T": -1,
        "V": -3,
        "W": -4,
        "Y": -3,
    },
    "E": {
        "A": -1,
        "C": -4,
        "D": 2,
        "E": 5,
        "F": -3,
        "G": -2,
        "H": 0,
        "I": -3,
        "K": 1,
        "L": -3,
        "M": -2,
        "N": 0,
        "P": -1,
        "Q": 2,
        "R": 0,
        "S": 0,
        "T": -1,
        "V": -2,
        "W": -3,
        "Y": -2,
    },
    "F": {
        "A": -2,
        "C": -2,
        "D": -3,
        "E": -3,
        "F": 6,
        "G": -3,
        "H": -1,
        "I": 0,
        "K": -3,
        "L": 0,
        "M": 0,
        "N": -3,
        "P": -4,
        "Q": -3,
        "R": -3,
        "S": -2,
        "T": -2,
        "V": -1,
        "W": 1,
        "Y": 3,
    },
    "G": {
        "A": 0,
        "C": -3,
        "D": -1,
        "E": -2,
        "F": -3,
        "G": 6,
        "H": -2,
        "I": -4,
        "K": -2,
        "L": -4,
        "M": -3,
        "N": 0,
        "P": -2,
        "Q": -2,
        "R": -2,
        "S": 0,
        "T": -2,
        "V": -3,
        "W": -2,
        "Y": -3,
    },
    "H": {
        "A": -2,
        "C": -3,
        "D": -1,
        "E": 0,
        "F": -1,
        "G": -2,
        "H": 8,
        "I": -3,
        "K": -1,
        "L": -3,
        "M": -2,
        "N": 1,
        "P": -2,
        "Q": 0,
        "R": 0,
        "S": -1,
        "T": -2,
        "V": -3,
        "W": -2,
        "Y": 2,
    },
    "I": {
        "A": -1,
        "C": -1,
        "D": -3,
        "E": -3,
        "F": 0,
        "G": -4,
        "H": -3,
        "I": 4,
        "K": -3,
        "L": 2,
        "M": 1,
        "N": -3,
        "P": -3,
        "Q": -3,
        "R": -3,
        "S": -2,
        "T": -1,
        "V": 3,
        "W": -3,
        "Y": -1,
    },
    "K": {
        "A": -1,
        "C": -3,
        "D": -1,
        "E": 1,
        "F": -3,
        "G": -2,
        "H": -1,
        "I": -3,
        "K": 5,
        "L": -2,
        "M": -1,
        "N": 0,
        "P": -1,
        "Q": 1,
        "R": 2,
        "S": 0,
        "T": -1,
        "V": -2,
        "W": -3,
        "Y": -2,
    },
    "L": {
        "A": -1,
        "C": -1,
        "D": -4,
        "E": -3,
        "F": 0,
        "G": -4,
        "H": -3,
        "I": 2,
        "K": -2,
        "L": 4,
        "M": 2,
        "N": -3,
        "P": -3,
        "Q": -2,
        "R": -2,
        "S": -2,
        "T": -1,
        "V": 1,
        "W": -2,
        "Y": -1,
    },
    "M": {
        "A": -1,
        "C": -1,
        "D": -3,
        "E": -2,
        "F": 0,
        "G": -3,
        "H": -2,
        "I": 1,
        "K": -1,
        "L": 2,
        "M": 5,
        "N": -2,
        "P": -2,
        "Q": 0,
        "R": -1,
        "S": -1,
        "T": -1,
        "V": 1,
        "W": -1,
        "Y": -1,
    },
    "N": {
        "A": -2,
        "C": -3,
        "D": 1,
        "E": 0,
        "F": -3,
        "G": 0,
        "H": 1,
        "I": -3,
        "K": 0,
        "L": -3,
        "M": -2,
        "N": 6,
        "P": -2,
        "Q": 0,
        "R": 0,
        "S": 1,
        "T": 0,
        "V": -3,
        "W": -4,
        "Y": -2,
    },
    "P": {
        "A": -1,
        "C": -3,
        "D": -1,
        "E": -1,
        "F": -4,
        "G": -2,
        "H": -2,
        "I": -3,
        "K": -1,
        "L": -3,
        "M": -2,
        "N": -2,
        "P": 7,
        "Q": -1,
        "R": -2,
        "S": -1,
        "T": -1,
        "V": -2,
        "W": -4,
        "Y": -3,
    },
    "Q": {
        "A": -1,
        "C": -3,
        "D": 0,
        "E": 2,
        "F": -3,
        "G": -2,
        "H": 0,
        "I": -3,
        "K": 1,
        "L": -2,
        "M": 0,
        "N": 0,
        "P": -1,
        "Q": 5,
        "R": 1,
        "S": 0,
        "T": -1,
        "V": -2,
        "W": -2,
        "Y": -1,
    },
    "R": {
        "A": -1,
        "C": -3,
        "D": -2,
        "E": 0,
        "F": -3,
        "G": -2,
        "H": 0,
        "I": -3,
        "K": 2,
        "L": -2,
        "M": -1,
        "N": 0,
        "P": -2,
        "Q": 1,
        "R": 5,
        "S": -1,
        "T": -1,
        "V": -3,
        "W": -3,
        "Y": -2,
    },
    "S": {
        "A": 1,
        "C": -1,
        "D": 0,
        "E": 0,
        "F": -2,
        "G": 0,
        "H": -1,
        "I": -2,
        "K": 0,
        "L": -2,
        "M": -1,
        "N": 1,
        "P": -1,
        "Q": 0,
        "R": -1,
        "S": 4,
        "T": 1,
        "V": -2,
        "W": -3,
        "Y": -2,
    },
    "T": {
        "A": 0,
        "C": -1,
        "D": -1,
        "E": -1,
        "F": -2,
        "G": -2,
        "H": -2,
        "I": -1,
        "K": -1,
        "L": -1,
        "M": -1,
        "N": 0,
        "P": -1,
        "Q": -1,
        "R": -1,
        "S": 1,
        "T": 5,
        "V": 0,
        "W": -2,
        "Y": -2,
    },
    "V": {
        "A": 0,
        "C": -1,
        "D": -3,
        "E": -2,
        "F": -1,
        "G": -3,
        "H": -3,
        "I": 3,
        "K": -2,
        "L": 1,
        "M": 1,
        "N": -3,
        "P": -2,
        "Q": -2,
        "R": -3,
        "S": -2,
        "T": 0,
        "V": 4,
        "W": -3,
        "Y": -1,
    },
    "W": {
        "A": -3,
        "C": -2,
        "D": -4,
        "E": -3,
        "F": 1,
        "G": -2,
        "H": -2,
        "I": -3,
        "K": -3,
        "L": -2,
        "M": -1,
        "N": -4,
        "P": -4,
        "Q": -2,
        "R": -3,
        "S": -3,
        "T": -2,
        "V": -3,
        "W": 11,
        "Y": 2,
    },
    "Y": {
        "A": -2,
        "C": -2,
        "D": -3,
        "E": -2,
        "F": 3,
        "G": -3,
        "H": 2,
        "I": -1,
        "K": -2,
        "L": -1,
        "M": -1,
        "N": -2,
        "P": -3,
        "Q": -1,
        "R": -2,
        "S": -2,
        "T": -2,
        "V": -1,
        "W": 2,
        "Y": 7,
    },
}


def _blosum_score(matrix, a, b, default=-4):
    try:
        return matrix[a][b]
    except Exception:
        return default


def choose_mutant_aa(
    orig_aa: str,
    policy: str,
    rng: np.random.Generator,
    blosum_name: str = "blosum62",
    blosum_temp: float = 1.0,
) -> str:
    policy = (policy or "alanine").lower()
    if policy == "alanine":
        return "A"
    if policy in ("to-x", "x"):
        return "X"
    if policy == "random":
        candidates = [aa for aa in STANDARD_AAS if aa != orig_aa]
        return rng.choice(candidates)
    if policy == "blosum":
        # Use embedded BLOSUM62
        matrix = BLOSUM62
        scores = np.array([_blosum_score(matrix, orig_aa, aa, default=-4) for aa in STANDARD_AAS])
        # Exclude self to ensure a change
        scores = scores.astype(float)
        for idx, aa in enumerate(STANDARD_AAS):
            if aa == orig_aa:
                scores[idx] = -np.inf
        # Convert to probabilities via softmax with temperature
        max_s = np.nanmax(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else 0.0
        logits = (scores - max_s) / max(1e-6, blosum_temp)
        # Handle -inf
        with np.errstate(over="ignore", invalid="ignore"):
            exp_logits = np.where(np.isfinite(logits), np.exp(logits), 0.0)
        probs = (
            exp_logits / exp_logits.sum()
            if exp_logits.sum() > 0
            else np.ones_like(exp_logits) / len(exp_logits)
        )
        choice = rng.choice(STANDARD_AAS, p=probs)
        return choice
    # Fallback: alanine
    return "A"


def representative_subsample_ids(sequences: dict, ids: list, n: int, seed: int = 42):
    """Select n representative sequence ids from ids based on dipeptide frequency diversity.

    Uses a greedy farthest-first traversal over L2 distance of normalized dipeptide frequency vectors.
    Deterministic given the same seed.
    """
    if len(ids) <= n:
        return ids

    aa_to_idx = {aa: i for i, aa in enumerate(STANDARD_AAS)}
    dim = len(STANDARD_AAS) * len(STANDARD_AAS)
    m = len(ids)
    X = np.zeros((m, dim), dtype=np.float32)
    lengths = np.zeros(m, dtype=int)

    for i, pid in enumerate(ids):
        seq = sequences.get(pid, "")
        lengths[i] = len(seq)
        # dipeptide counts
        for j in range(len(seq) - 1):
            a = seq[j]
            b = seq[j + 1]
            if a in aa_to_idx and b in aa_to_idx:
                idx = aa_to_idx[a] * len(STANDARD_AAS) + aa_to_idx[b]
                X[i, idx] += 1.0
        # L2 normalize if nonzero
        norm = np.linalg.norm(X[i])
        if norm > 0:
            X[i] /= norm

    # initialize with the longest sequence to prefer full-length representatives
    first = int(np.argmax(lengths))
    selected = [first]
    remaining = set(range(m)) - {first}

    # precompute squared norms for fast distances
    # greedy farthest-first
    while len(selected) < n and remaining:
        sel_mat = X[selected]
        # compute min distance to selected for each remaining
        rem_idx = np.array(sorted(remaining), dtype=int)
        rem_mat = X[rem_idx]
        # compute squared distances efficiently: ||u-v||^2 = ||u||^2 + ||v||^2 - 2u.v
        # precompute norms
        sel_norms = np.sum(sel_mat * sel_mat, axis=1)
        rem_norms = np.sum(rem_mat * rem_mat, axis=1)
        # dot products
        dots = rem_mat.dot(sel_mat.T)
        # compute min distances
        dists = np.minimum.reduce([rem_norms[:, None] + sel_norms[None, :] - 2 * dots])
        min_dists = np.min(dists, axis=1)
        # pick index with max min_dist
        pick_idx = int(np.argmax(min_dists))
        pick = rem_idx[pick_idx]
        selected.append(pick)
        remaining.remove(pick)

    # map back to ids
    selected_ids = [ids[i] for i in selected[:n]]
    return selected_ids


def create_cluster_heatmaps(
    df,
    protein_clusters,
    sequences,
    cluster_name,
    model_type,
    output_prefix,
    sdp_dict=None,
    args=None,
):
    """Create heatmaps for all proteins in a cluster with proportional widths."""
    cluster_proteins = protein_clusters[protein_clusters["predicted_class"] == cluster_name][
        "protein_id"
    ].tolist()

    if not cluster_proteins:
        print(f"No proteins found for cluster: {cluster_name}")
        return

    # Get actual residue data lengths (not sequence lengths)
    residue_lengths = []
    valid_proteins = []
    for protein_id in cluster_proteins:
        protein_data = df[(df["protein_id"] == protein_id) & (df["class"] == cluster_name)]
        if not protein_data.empty:
            residue_count = len(protein_data["residue"].unique())
            residue_lengths.append(residue_count)
            valid_proteins.append(protein_id)

    if not valid_proteins:
        print(f"No valid proteins with data for cluster: {cluster_name}")
        return

    pdf_path = f"{output_prefix}_cluster_heatmaps_{cluster_name}.pdf"

    with PdfPages(pdf_path) as pdf:
        proteins_per_page = 12  # Reduced to accommodate proper spacing

        for page_start in range(0, len(valid_proteins), proteins_per_page):
            page_proteins = valid_proteins[page_start : page_start + proteins_per_page]
            page_residue_lengths = residue_lengths[page_start : page_start + proteins_per_page]
            max_residue_len = max(page_residue_lengths)

            fig = plt.figure(figsize=(8.27, 11.69))  # A4 size

            for i, protein_id in enumerate(page_proteins):
                residue_len = page_residue_lengths[i]

                # Get ablation data with actual residue positions
                protein_data = df[(df["protein_id"] == protein_id) & (df["class"] == cluster_name)]
                if protein_data.empty:
                    continue

                residue_data = protein_data.set_index("residue")["ablation_score"]
                residues = sorted(residue_data.index)
                heat_values = [residue_data[r] for r in residues]

                # Layout with proportional width
                max_strip_width = 0.7  # 70% of page width for longest protein
                strip_width = max_strip_width * (residue_len / max_residue_len)
                strip_height = 0.02  # Fixed height (~5mm)
                start_x = 0.1
                y_spacing = 0.08  # Increased spacing to prevent overlap
                start_y = 0.85 - (i * y_spacing)

                # Add protein name ABOVE the strip with proper clearance
                fig.text(
                    start_x,
                    start_y + 0.025,
                    f"{protein_id} (residues {min(residues)}-{max(residues)})",
                    fontsize=9,
                    verticalalignment="bottom",
                )

                # Create axis for heatmap
                ax = fig.add_axes([start_x, start_y, strip_width, strip_height])
                im = ax.imshow([heat_values], cmap="coolwarm", aspect="auto")

                ax.set_ylabel("")
                ax.set_yticks([])

                # Correct sequence position ticks
                n_ticks = 5
                tick_indices = [i * (len(residues) - 1) // (n_ticks - 1) for i in range(n_ticks)]
                tick_positions = tick_indices  # Position in heatmap array
                tick_labels = [residues[i] for i in tick_indices]  # Actual residue numbers

                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, fontsize=7)

                # Mark SDPs if provided
                if sdp_dict and protein_id in sdp_dict:
                    sdp_res = sdp_dict[protein_id]
                    for idx, res in enumerate(residues):
                        if res in sdp_res:
                            # Draw a small triangle above the strip (outside axis), do not expand box
                            ax.plot(
                                idx,
                                -0.4,
                                marker=(3, 0, 180),
                                color="black",
                                markersize=3,
                                clip_on=False,
                            )

            fig.suptitle(
                f"{cluster_name} Cluster - Proportional Width Heatmaps (Page {page_start//proteins_per_page + 1})",
                fontsize=14,
                y=0.98,
            )
            pdf.savefig(fig, bbox_inches="tight", dpi=200)
            plt.close(fig)

    print(f"Created {pdf_path} with {len(valid_proteins)} proteins")


def create_cluster_msa_with_ablation_coloring(
    df,
    protein_clusters,
    sequences,
    cluster_name,
    model_type,
    output_prefix,
    sdp_dict=None,
    args=None,
):
    """Create MSA for cluster with residues colored by ablation scores."""
    cluster_proteins = protein_clusters[protein_clusters["predicted_class"] == cluster_name][
        "protein_id"
    ].tolist()

    if not cluster_proteins:
        print(f"No proteins found for cluster: {cluster_name}")
        return

    # Get sequences for this cluster
    cluster_sequences = {}
    for protein_id in cluster_proteins:
        if protein_id in sequences:
            cluster_sequences[protein_id] = sequences[protein_id]

    if len(cluster_sequences) < 2:
        print(f"Need at least 2 sequences for MSA in cluster: {cluster_name}")
        return

    print(f"Creating MSA for {cluster_name} cluster with {len(cluster_sequences)} proteins")

    # Decide whether to subsample before creating FASTA for MAFFT
    subsampled_ids = None
    if args is not None:
        max_seqs = getattr(args, "max_msa_seqs", 100)
        subsample_enabled = getattr(args, "subsample_msa", False)
        seed = getattr(args, "random_seed", 42)
    else:
        max_seqs = 100
        subsample_enabled = False
        seed = 42

    all_ids = list(cluster_sequences.keys())
    if subsample_enabled and len(all_ids) > max_seqs:
        # Estimate alignment length as median sequence length of cluster
        seq_lengths = [len(cluster_sequences[p]) for p in all_ids]
        est_len = int(np.median(seq_lengths)) if seq_lengths else 0
        max_total = getattr(args, "max_total_residues", 50000) if args is not None else 50000
        if est_len > 0:
            max_proteins_by_residues = max(2, int(max_total // est_len))
        else:
            max_proteins_by_residues = max_seqs

        target_n = min(max_seqs, max_proteins_by_residues)
        target_n = max(2, target_n)

        subsampled_ids = representative_subsample_ids(cluster_sequences, all_ids, target_n, seed)
        print(
            f"Cluster {cluster_name} has {len(all_ids)} sequences; estimated alignment len={est_len}; subsampling to {len(subsampled_ids)} sequences for MSA/plotting (target_n={target_n})"
        )
    else:
        subsampled_ids = all_ids

    # Create temporary FASTA file for MAFFT containing only subsampled sequences
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as temp_fasta:
        for protein_id in subsampled_ids:
            seq = cluster_sequences[protein_id]
            temp_fasta.write(f">{protein_id}\n{seq}\n")
        temp_fasta_path = temp_fasta.name

    try:
        # Run MAFFT alignment
        output_file = f"{output_prefix}_msa_{cluster_name}_alignment.fasta"
        cmd = ["mafft", "--auto", temp_fasta_path]

        with open(output_file, "w") as outf:
            result = subprocess.run(cmd, stdout=outf, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f"MAFFT failed for {cluster_name}: {result.stderr}")
            return

        print(f"MSA completed for {cluster_name}, saved to {output_file}")

        # Parse alignment
        alignment = AlignIO.read(output_file, "fasta")
        alignment_length = alignment.get_alignment_length()

        # Create ablation score mapping for each protein
        protein_ablation_scores = {}
        for protein_id in subsampled_ids:
            protein_data = df[(df["protein_id"] == protein_id) & (df["class"] == cluster_name)]
            if not protein_data.empty:
                scores = protein_data.set_index("residue")["ablation_score"]
                protein_ablation_scores[protein_id] = scores.to_dict()

        # Create colored MSA visualization
        create_colored_msa_plot(
            alignment,
            protein_ablation_scores,
            cluster_name,
            model_type,
            output_prefix,
            sdp_dict,
            args,
        )

    finally:
        # Clean up temporary file
        os.unlink(temp_fasta_path)


def create_colored_msa_plot(
    alignment,
    protein_ablation_scores,
    cluster_name,
    model_type,
    output_prefix,
    sdp_dict=None,
    args=None,
):
    """Create a colored MSA plot with ablation scores - rewritten for memory efficiency."""
    num_seqs = len(alignment)
    align_len = alignment.get_alignment_length()

    # Configurable thresholds for skipping large MSAs (can be overridden by --force-plots)
    max_seqs = None
    max_len = None
    force = False
    if args is not None:
        max_seqs = getattr(args, "max_msa_seqs", None)
        max_len = getattr(args, "max_msa_length", None)
        force = getattr(args, "force_plots", False)

    # Use defaults if not provided
    if max_seqs is None:
        max_seqs = 100
    if max_len is None:
        max_len = 2000

    if not force and (num_seqs > max_seqs or align_len > max_len):
        print(
            f"Skipping MSA visualization for {cluster_name} - too large ({num_seqs} sequences, {align_len} positions)."
            f" Thresholds: max_seqs={max_seqs}, max_len={max_len}. Use --force-plots to override."
        )
        return

    # Calculate reasonable figure size - make squares bigger to fit letters
    fig_width = min(max(10, align_len * 0.3), 24)  # Bigger squares for letters
    fig_height = min(max(6, num_seqs * 0.4), 16)  # Bigger squares for letters

    print(
        f"Creating MSA plot for {cluster_name}: {num_seqs} seqs x {align_len} positions, figure size: {fig_width:.1f}x{fig_height:.1f}"
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use even lower DPI for all MSA plots to be safe
    save_dpi = 100

    # Color map for ablation scores (coolwarm to match heatmaps)
    from matplotlib.cm import coolwarm
    from matplotlib.colors import Normalize

    # Normalize scores per protein
    normalized_protein_scores = {}
    all_normalized_scores = []

    for protein_id, scores_dict in protein_ablation_scores.items():
        if scores_dict:
            scores_array = np.array(list(scores_dict.values()))

            # Per-protein normalization (z-score normalization)
            if len(scores_array) > 1 and np.std(scores_array) > 0:
                normalized_scores = (scores_array - np.mean(scores_array)) / np.std(scores_array)
            else:
                normalized_scores = scores_array - np.mean(scores_array)

            # Create normalized dictionary
            normalized_dict = {}
            for i, (orig_pos, _) in enumerate(scores_dict.items()):
                normalized_dict[orig_pos] = normalized_scores[i]

            normalized_protein_scores[protein_id] = normalized_dict
            all_normalized_scores.extend(normalized_scores)

    if all_normalized_scores:
        # Use symmetric normalization around 0 for coolwarm
        abs_max = max(abs(np.min(all_normalized_scores)), abs(np.max(all_normalized_scores)))
        vmin = -abs_max if abs_max > 0 else -1
        vmax = abs_max if abs_max > 0 else 1
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = coolwarm
    else:
        norm = Normalize(vmin=-1, vmax=1)
        colormap = coolwarm

    # Memory-efficient visualization: draw rectangles individually instead of large matrices
    for seq_idx in range(num_seqs):
        record = alignment[seq_idx]
        protein_id = record.id
        seq_str = str(record.seq)

        # For SDP marking
        sdp_res = set()
        if sdp_dict and protein_id in sdp_dict:
            sdp_res = sdp_dict[protein_id]

        orig_pos_counter = 0
        for pos in range(align_len):
            aa = seq_str[pos] if pos < len(seq_str) else "-"

            # Default grey color for gaps
            if aa == "-":
                color = [0.9, 0.9, 0.9, 1.0]
            else:
                orig_pos_counter += 1
                color = [0.95, 0.95, 0.95, 1.0]  # Default light grey

                # Color based on ablation score if available
                if protein_id in normalized_protein_scores:
                    if orig_pos_counter in normalized_protein_scores[protein_id]:
                        score = normalized_protein_scores[protein_id][orig_pos_counter]
                        color = colormap(norm(score))

            # Draw the cell as a rectangle
            rect = plt.Rectangle(
                (pos - 0.5, seq_idx - 0.5),
                1,
                1,
                facecolor=color,
                edgecolor="lightgray",
                linewidth=0.1,
            )
            ax.add_patch(rect)

            # Mark SDP with a black rectangle border (no triangles)
            if aa != "-" and sdp_res and orig_pos_counter in sdp_res:
                border_rect = plt.Rectangle(
                    (pos - 0.5, seq_idx - 0.5),
                    1,
                    1,
                    linewidth=0.3,
                    edgecolor="black",
                    facecolor="none",
                    zorder=10,
                )
                ax.add_patch(border_rect)

    # Add text labels for amino acids - always include letters
    font_size = min(8, max(2, 80 / max(align_len, num_seqs)))  # Better scaling for longer MSAs

    # Always add text, but adjust font size based on alignment size
    for seq_idx, record in enumerate(alignment):
        seq_str = str(record.seq)
        protein_id = record.id

        # For SDP marking
        sdp_res = set()
        if sdp_dict and protein_id in sdp_dict:
            sdp_res = sdp_dict[protein_id]

        orig_pos_counter = 0
        for pos, aa in enumerate(seq_str):
            if aa != "-":
                orig_pos_counter += 1

            # Determine text color based on background
            text_color = "black"
            if aa != "-" and protein_id in normalized_protein_scores:
                if orig_pos_counter in normalized_protein_scores[protein_id]:
                    score = normalized_protein_scores[protein_id][orig_pos_counter]
                    try:
                        color_intensity = norm(score)
                        text_color = (
                            "white" if (color_intensity < 0.3 or color_intensity > 0.7) else "black"
                        )
                    except:
                        text_color = "black"

            # Add the amino acid text - always include
            ax.text(
                pos,
                seq_idx,
                aa,
                ha="center",
                va="center",
                fontsize=font_size,
                color=text_color,
                weight="bold",
                fontfamily="monospace",
            )

    # Set axis properties
    ax.set_xlim(-0.5, align_len - 0.5)
    ax.set_ylim(-0.5, num_seqs - 0.5)
    ax.set_aspect("equal")

    # Add sequence labels (smaller font size for protein names)
    seq_labels = [record.id for record in alignment]
    ax.set_yticks(range(num_seqs))
    label_fontsize = min(6, max(3, 60 / num_seqs))  # Smaller font for protein names
    ax.set_yticklabels(seq_labels, fontsize=label_fontsize)

    # Add position labels (reduce frequency for long alignments)
    tick_step = max(1, align_len // 15)  # At most 15 ticks
    tick_positions = range(0, align_len, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(i + 1) for i in tick_positions], fontsize=6)

    ax.set_title(f"{cluster_name} Cluster MSA", fontsize=12, pad=10)

    plt.tight_layout()

    # Save the plot
    output_path = f"{output_prefix}_msa_{cluster_name}_colored_ablation.pdf"
    plt.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
    plt.close()

    print(f"Colored MSA plot saved to {output_path}")


# --- Main ablation logic ---
def ablation_scores_for_protein(
    seq_id,
    sequence,
    model,
    probe,
    window_size,
    scan_mode="ablation",
    mutation_policy: str = "alanine",
    random_seed: int = 42,
    blosum_name: str = "blosum62",
    blosum_temp: float = 1.0,
    progress: bool = True,
):
    # Use new standardized interface for residue and mean embeddings
    if not (hasattr(model, "get_residue_embeddings") and hasattr(model, "get_mean_embedding")):
        raise ValueError(
            "Model must implement get_residue_embeddings and get_mean_embedding methods."
        )
    seq_len = len(sequence)
    # Full mean-pooled embedding
    full_emb = model.get_mean_embedding(sequence, seq_id)
    # Get full sequence cluster probabilities
    full_emb_norm = probe.normalize_embedding(full_emb.reshape(1, -1))
    full_probs = probe.model.predict_proba(full_emb_norm)[0]
    # For each residue/window, score according to scan_mode
    ablation_scores = np.zeros((seq_len, len(full_probs)))
    ablation_details = [{} for _ in range(seq_len)]
    most_probable_cluster = np.argmax(full_probs)
    full_pred = most_probable_cluster
    penalty = 100.0  # Large penalty for cluster change

    if scan_mode == "ablation":
        # Use residue embeddings and mean-pool with masked window
        residue_emb = model.get_residue_embeddings(sequence, seq_id)
        residue_seq_len, emb_dim = residue_emb.shape
        actual_seq_len = seq_len
        if residue_seq_len != actual_seq_len:
            print(
                f"WARNING: {seq_id} - sequence length {actual_seq_len} != residue embeddings length {residue_seq_len}"
            )
            if residue_seq_len < actual_seq_len:
                padding = np.zeros((actual_seq_len - residue_seq_len, emb_dim))
                residue_emb = np.vstack([residue_emb, padding])
            else:
                residue_emb = residue_emb[:actual_seq_len]

        # Progress setup
        report_every = max(1, seq_len // 10)
        for start in range(seq_len):
            end = min(start + window_size, seq_len)
            mask = np.ones(seq_len, dtype=bool)
            mask[start:end] = False
            ablated_emb = mean_pool_embedding(residue_emb, mask)
            ablated_emb_norm = probe.normalize_embedding(ablated_emb.reshape(1, -1))
            ablated_probs = probe.model.predict_proba(ablated_emb_norm)[0]
            ablated_pred = np.argmax(ablated_probs)

            if progress and ((start + 1) % report_every == 0 or start == 0 or start == seq_len - 1):
                pct = int(((start + 1) / seq_len) * 100)
                sys.stdout.write(
                    f"\r  [{seq_id}] {scan_mode}: {start + 1}/{seq_len} residues ({pct}%)"
                )
                sys.stdout.flush()

            # If cluster assignment changes, add a large penalty
            if ablated_pred != full_pred:
                for i in range(start, end):
                    ablation_scores[i, most_probable_cluster] += penalty
                    ablation_details[i] = {
                        "Pasigned_full": full_probs[most_probable_cluster],
                        "Pasigned_ablated": ablated_probs[most_probable_cluster],
                        "Pclosest_full": np.nan,
                        "Pclosest_ablated": np.nan,
                    }
            else:
                # Margin for full embedding
                sorted_full = np.argsort(full_probs)[::-1]
                competitor_full = (
                    sorted_full[1] if sorted_full[0] == most_probable_cluster else sorted_full[0]
                )
                margin_full = full_probs[most_probable_cluster] - full_probs[competitor_full]

                # Margin for ablated embedding
                sorted_ablated = np.argsort(ablated_probs)[::-1]
                competitor_ablated = (
                    sorted_ablated[1]
                    if sorted_ablated[0] == most_probable_cluster
                    else sorted_ablated[0]
                )
                margin_ablated = (
                    ablated_probs[most_probable_cluster] - ablated_probs[competitor_ablated]
                )

                # Score is the difference in margin (full - ablated)
                margin_diff = margin_full - margin_ablated
                for i in range(start, end):
                    ablation_scores[i, most_probable_cluster] += margin_diff
                    ablation_details[i] = {
                        "Pasigned_full": full_probs[most_probable_cluster],
                        "Pasigned_ablated": ablated_probs[most_probable_cluster],
                        "Pclosest_full": full_probs[competitor_full],
                        "Pclosest_ablated": ablated_probs[competitor_ablated],
                    }
    elif scan_mode == "ala-scanning":
        # Replace a single residue at a time using a mutation policy and recompute mean-pooled embedding
        rng = np.random.default_rng(random_seed)
        report_every = max(1, seq_len // 10)
        for i in range(seq_len):
            # Choose mutant AA according to policy (default alanine)
            mutant_aa = choose_mutant_aa(
                sequence[i], mutation_policy, rng, blosum_name, blosum_temp
            )
            mutated_seq = sequence[:i] + mutant_aa + sequence[i + 1 :]

            mut_emb = model.get_mean_embedding(mutated_seq, f"{seq_id}_ala_{i+1}")
            mut_emb_norm = probe.normalize_embedding(mut_emb.reshape(1, -1))
            ablated_probs = probe.model.predict_proba(mut_emb_norm)[0]
            ablated_pred = np.argmax(ablated_probs)

            if ablated_pred != full_pred:
                ablation_scores[i, most_probable_cluster] += penalty
                ablation_details[i] = {
                    "Pasigned_full": full_probs[most_probable_cluster],
                    "Pasigned_ablated": ablated_probs[most_probable_cluster],
                    "Pclosest_full": np.nan,
                    "Pclosest_ablated": np.nan,
                    "mutant_aa": mutant_aa,
                }
            else:
                sorted_full = np.argsort(full_probs)[::-1]
                competitor_full = (
                    sorted_full[1] if sorted_full[0] == most_probable_cluster else sorted_full[0]
                )
                margin_full = full_probs[most_probable_cluster] - full_probs[competitor_full]

                sorted_ablated = np.argsort(ablated_probs)[::-1]
                competitor_ablated = (
                    sorted_ablated[1]
                    if sorted_ablated[0] == most_probable_cluster
                    else sorted_ablated[0]
                )
                margin_ablated = (
                    ablated_probs[most_probable_cluster] - ablated_probs[competitor_ablated]
                )
                margin_diff = margin_full - margin_ablated
                ablation_scores[i, most_probable_cluster] += margin_diff
                ablation_details[i] = {
                    "Pasigned_full": full_probs[most_probable_cluster],
                    "Pasigned_ablated": ablated_probs[most_probable_cluster],
                    "Pclosest_full": full_probs[competitor_full],
                    "Pclosest_ablated": ablated_probs[competitor_ablated],
                    "mutant_aa": mutant_aa,
                }

            if progress and ((i + 1) % report_every == 0 or i == 0 or i == seq_len - 1):
                pct = int(((i + 1) / seq_len) * 100)
                sys.stdout.write(f"\r  [{seq_id}] {scan_mode}: {i + 1}/{seq_len} residues ({pct}%)")
                sys.stdout.flush()
    else:
        raise ValueError(f"Unknown scan_mode: {scan_mode}")
    return ablation_scores, full_probs, ablation_details


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Residue ablation probe for protein cluster assignment interpretation"
    )
    parser.add_argument("--input", required=True, help="Input protein file (TSV or FASTA)")
    parser.add_argument("--input_type", choices=["tsv", "fasta"], default="tsv")
    parser.add_argument("--id_column", default="uniprot_id", help="ID column for TSV input")
    parser.add_argument("--seq_column", default="sequence", help="Sequence column for TSV input")
    parser.add_argument(
        "--model_type", required=True, help="PLM model type (as in generate_embeddings.py)"
    )
    parser.add_argument("--model_name", help="Model name/path (model-specific)")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--probe_model", required=True, help="Trained linear probe model (pickle)")
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Window size L for ablation (default: 5). Ignored for --scan_mode ala-scanning",
    )
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files")
    parser.add_argument(
        "--scan_mode",
        choices=["ablation", "ala-scanning"],
        default="ablation",
        help="Scanning mode: ablation (mask window and mean-pool) or ala-scanning (per-residue mutation using a policy)",
    )
    parser.add_argument(
        "--mutation_policy",
        choices=["alanine", "to-x", "random", "blosum"],
        default="alanine",
        help="Policy for selecting the replacement residue when scan_mode=ala-scanning",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed used for stochastic mutation policies",
    )
    parser.add_argument(
        "--blosum_name",
        choices=["blosum62"],
        default="blosum62",
        help="Which BLOSUM matrix to use when mutation_policy=blosum",
    )
    parser.add_argument(
        "--blosum_temp",
        type=float,
        default=1.0,
        help="Temperature for softmax sampling from BLOSUM scores (lower=sharper)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators",
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of parallel threads to use for protein processing (default: 1 for serial processing)",
    )
    parser.add_argument(
        "--model_args",
        nargs="*",
        default=[],
        help="Additional model-specific arguments (key=value pairs)",
    )
    parser.add_argument(
        "--sdp_tsv",
        type=str,
        default=None,
        help="Optional: Path to SDP TSV file (first column: protein ids, next columns: residue coordinates)",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip proportional-width heatmap plots (cluster heatmaps)",
    )
    parser.add_argument(
        "--no-alignment-plots",
        action="store_true",
        help="Skip MSA / alignment plots (colored MSAs)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip all plotting/visualizations (equivalent to --no-strip-plots and --no-alignment-plots)",
    )
    parser.add_argument(
        "--max-clusters-to-plot",
        type=int,
        default=None,
        help="Maximum number of clusters to generate plots for (None = all)",
    )
    parser.add_argument(
        "--max-msa-seqs",
        type=int,
        default=100,
        help="Maximum number of sequences in an MSA before skipping the colored MSA plot (default: 100)",
    )
    parser.add_argument(
        "--max-msa-length",
        type=int,
        default=2000,
        help="Maximum MSA alignment length (positions) before skipping colored MSA plot (default: 2000)",
    )
    parser.add_argument(
        "--force-plots",
        action="store_true",
        help="Force generation of plots even if they exceed safety thresholds",
    )
    parser.add_argument(
        "--subsample-msa",
        action="store_true",
        help="If set and the cluster has more sequences than --max-msa-seqs, subsample to --max-msa-seqs for MSA+plotting",
    )
    parser.add_argument(
        "--max-total-residues",
        type=int,
        default=50000,
        help="Maximum total residues for MSA plotting (estimated alignment length * num_proteins). Subsampling will keep proteins so product <= this (default: 50000)",
    )
    # Normalization options (match linear_probe.py)
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Skip feature standardization (center=False, scale=False)",
    )
    parser.add_argument(
        "--norm-center",
        action="store_true",
        default=True,
        help="Center features (subtract mean)",
    )
    parser.add_argument(
        "--no-norm-center",
        dest="norm_center",
        action="store_false",
        help="Don't center features",
    )
    parser.add_argument(
        "--norm-scale",
        action="store_true",
        default=True,
        help="Scale features (divide by std)",
    )
    parser.add_argument(
        "--no-norm-scale",
        dest="norm_scale",
        action="store_false",
        help="Don't scale features",
    )
    parser.add_argument(
        "--norm-pca-components",
        type=float,
        default=0.95,
        help="Number of PCA components or variance to retain (default: 0.95 for 95%% variance)",
    )
    parser.add_argument(
        "--norm-l2",
        action="store_true",
        default=True,
        help="Apply L2 normalization",
    )
    parser.add_argument(
        "--no-norm-l2",
        dest="norm_l2",
        action="store_false",
        help="Don't apply L2 normalization",
    )
    return parser.parse_args()


def load_input_sequences(args):
    """Load sequences from input file, filtering out invalid entries."""
    if args.input_type == "fasta":
        return load_sequences_from_fasta(args.input)
    else:
        # Load sequences and filter out NaN/missing values
        sequences = load_sequences_from_tsv(args.input, args.id_column, args.seq_column)

        # Filter out invalid sequences
        valid_sequences = {}
        for seq_id, seq in sequences.items():
            if pd.isna(seq) or pd.isna(seq_id) or seq == "NA" or seq_id == "NA":
                print(
                    f"Warning: Skipping sequence with invalid data - ID: {seq_id}, sequence: {seq}"
                )
                continue
            if not isinstance(seq, str) or len(seq) == 0:
                print(
                    f"Warning: Skipping sequence with invalid format - ID: {seq_id}, sequence type: {type(seq)}"
                )
                continue
            valid_sequences[seq_id] = seq

        print(f"Loaded {len(valid_sequences)} valid sequences out of {len(sequences)} total")
        return valid_sequences


def parse_model_arguments(args):
    """Parse model-specific arguments."""
    model_kwargs = {}
    for arg in args.model_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Try to convert to appropriate type
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
            model_kwargs[key] = value
    return model_kwargs


def setup_device(args):
    """Setup device for model inference."""
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(args.device)


def load_and_setup_model(args, model_kwargs, device):
    """Load and setup the protein language model."""
    model_class = load_model_class(args.model_type)
    model = model_class(model_name=args.model_name, device=device, **model_kwargs)
    model.load_model()
    return model


def setup_probe_wrapper(probe_obj, args):
    """Setup probe wrapper with normalization functions."""
    # Legacy: dict or class
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import normalize as l2_normalize

    def normalize_embedding(x):
        norm_info = getattr(probe_obj, "normalization_info", None)
        if norm_info is not None:
            normed = x
            scaler = norm_info.get("scaler", None)
            if scaler is not None:
                normed = scaler.transform(normed)
            pca = norm_info.get("pca", None)
            if pca is not None:
                normed = pca.transform(normed)
            if norm_info.get("l2", False):
                normed = l2_normalize(normed)
            return normed
        # Otherwise, use CLI options (legacy/fallback)
        if args.no_standardize:
            normed = x
        else:
            if args.norm_center or args.norm_scale:
                scaler = StandardScaler(with_mean=args.norm_center, with_std=args.norm_scale)
                normed = scaler.fit_transform(x)
            else:
                normed = x
        if (
            args.norm_pca_components
            and args.norm_pca_components > 0
            and normed.shape[1] > 1
            and normed.shape[0] > 1
        ):
            pca = PCA(n_components=args.norm_pca_components)
            normed = pca.fit_transform(normed)
        if args.norm_l2:
            normed = l2_normalize(normed)
        return normed

    class ProbeWrapper:
        def __init__(self, probe_dict):
            self.model = probe_dict["model"] if "model" in probe_dict else probe_dict.get("clf")
            self.class_names = probe_dict.get(
                "class_names", [str(i) for i in range(self.model.coef_.shape[0])]
            )
            self.normalization_info = probe_dict.get("normalization_info", None)

        def normalize_embedding(self, x):
            return normalize_embedding(x)

    if isinstance(probe_obj, dict):
        return ProbeWrapper(probe_obj)
    else:
        probe_obj.normalize_embedding = normalize_embedding
        return probe_obj


def load_and_setup_probe(args):
    """Load and setup the linear probe model."""
    probe_obj = load_linear_probe_model(args.probe_model)

    # If probe_obj is a dict and has 'model', treat as all-in-one package
    if isinstance(probe_obj, dict) and "model" in probe_obj:

        class ProbeAllInOne:
            def __init__(self, d):
                self.model = d["model"]
                self.class_names = d.get(
                    "class_names", [str(i) for i in range(self.model.coef_.shape[0])]
                )
                self.normalization_info = d.get("normalization_info", None)

            def normalize_embedding(self, x):
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                from sklearn.preprocessing import normalize as l2_normalize

                norm_info = self.normalization_info
                if norm_info is not None:
                    normed = x
                    # Check for scaler (key could be 'scaler')
                    scaler = norm_info.get("scaler", None)
                    if scaler is not None:
                        normed = scaler.transform(normed)
                    # Check for PCA (key could be 'pca' or 'pca_transform')
                    pca = norm_info.get("pca", None) or norm_info.get("pca_transform", None)
                    if pca is not None:
                        normed = pca.transform(normed)
                    # Check for L2 normalization (key could be 'l2' or 'l2_normalization')
                    l2_norm = norm_info.get("l2", False) or norm_info.get("l2_normalization", False)
                    if l2_norm:
                        normed = l2_normalize(normed)
                    return normed
                # fallback: no normalization info
                return x

        return ProbeAllInOne(probe_obj)
    else:
        return setup_probe_wrapper(probe_obj, args)


def process_single_protein(protein_data, model, probe, args):
    """
    Process a single protein for ablation scores.

    Args:
        protein_data: tuple of (seq_id, sequence, protein_index, total_proteins)
        model: protein language model
        probe: trained probe model
        args: command line arguments

    Returns:
        list of score dictionaries for this protein
    """
    seq_id, seq, protein_idx, total_proteins = protein_data

    class_names = (
        probe.class_names
        if hasattr(probe, "class_names")
        else [str(i) for i in range(probe.model.coef_.shape[0])]
    )

    if not getattr(args, "no_progress", False):
        print(
            f"Processing protein {protein_idx}/{total_proteins}: {seq_id} (len={len(seq)})",
            flush=True,
        )

    ablation_scores, full_probs, ablation_details = ablation_scores_for_protein(
        seq_id,
        seq,
        model,
        probe,
        args.window_size,
        args.scan_mode,
        args.mutation_policy,
        args.random_seed,
        args.blosum_name,
        args.blosum_temp,
        progress=(not getattr(args, "no_progress", False)),
    )

    protein_scores = []
    for i, aa in enumerate(seq):
        for c, cname in enumerate(class_names):
            score = ablation_scores[i, c]
            # Only output rows with nonzero score
            if score != 0:
                details = ablation_details[i]
                protein_scores.append(
                    {
                        "protein_id": seq_id,
                        "residue": i + 1,  # 1-based position
                        "aa": aa,
                        "mutant_aa": details.get("mutant_aa", np.nan),
                        "class": cname,
                        "ablation_score": score,  # 0-based index
                        "Pasigned_full": details.get("Pasigned_full", np.nan),
                        "Pasigned_ablated": details.get("Pasigned_ablated", np.nan),
                        "Pclosest_full": details.get("Pclosest_full", np.nan),
                        "Pclosest_ablated": details.get("Pclosest_ablated", np.nan),
                    }
                )

    if not getattr(args, "no_progress", False):
        # Finish any in-line progress line from per-residue loop
        print("")

    return protein_scores


def compute_all_ablation_scores(sequences, model, probe, args):
    """Compute ablation scores for all proteins, optionally in parallel."""
    n_processes = getattr(args, "n_processes", 1)

    if n_processes == 1:
        # Serial processing (original behavior)
        return compute_all_ablation_scores_serial(sequences, model, probe, args)
    else:
        # Parallel processing
        return compute_all_ablation_scores_parallel(sequences, model, probe, args, n_processes)


def compute_all_ablation_scores_serial(sequences, model, probe, args):
    """Compute ablation scores for all proteins serially (original behavior)."""
    class_names = (
        probe.class_names
        if hasattr(probe, "class_names")
        else [str(i) for i in range(probe.model.coef_.shape[0])]
    )

    all_scores = []
    total = len(sequences)
    for idx, (seq_id, seq) in enumerate(sequences.items(), start=1):
        if not getattr(args, "no_progress", False):
            print(f"Processing protein {idx}/{total}: {seq_id} (len={len(seq)})", flush=True)
        ablation_scores, full_probs, ablation_details = ablation_scores_for_protein(
            seq_id,
            seq,
            model,
            probe,
            args.window_size,
            args.scan_mode,
            args.mutation_policy,
            args.random_seed,
            args.blosum_name,
            args.blosum_temp,
            progress=(not getattr(args, "no_progress", False)),
        )
        if not getattr(args, "no_progress", False):
            # Finish any in-line progress line from per-residue loop
            print("")
        for i, aa in enumerate(seq):
            for c, cname in enumerate(class_names):
                score = ablation_scores[i, c]
                # Only output rows with nonzero score
                if score != 0:
                    details = ablation_details[i]
                    all_scores.append(
                        {
                            "protein_id": seq_id,
                            "residue": i + 1,  # 1-based position
                            "aa": aa,
                            "mutant_aa": details.get("mutant_aa", np.nan),
                            "class": cname,
                            "ablation_score": score,  # 0-based index
                            "Pasigned_full": details.get("Pasigned_full", np.nan),
                            "Pasigned_ablated": details.get("Pasigned_ablated", np.nan),
                            "Pclosest_full": details.get("Pclosest_full", np.nan),
                            "Pclosest_ablated": details.get("Pclosest_ablated", np.nan),
                        }
                    )
    return pd.DataFrame(all_scores)


def process_single_protein_shared(protein_data, model, probe, args):
    """
    Process a single protein for ablation scores using shared model and probe.

    Args:
        protein_data: tuple of (seq_id, sequence, protein_index, total_proteins)
        model: shared protein language model
        probe: shared trained probe model
        args: command line arguments

    Returns:
        list of score dictionaries for this protein
    """
    seq_id, seq, protein_idx, total_proteins = protein_data

    class_names = (
        probe.class_names
        if hasattr(probe, "class_names")
        else [str(i) for i in range(probe.model.coef_.shape[0])]
    )

    if not getattr(args, "no_progress", False):
        # Thread-safe print
        print(
            f"Processing protein {protein_idx}/{total_proteins}: {seq_id} (len={len(seq)})",
            flush=True,
        )

    ablation_scores, full_probs, ablation_details = ablation_scores_for_protein(
        seq_id,
        seq,
        model,
        probe,
        args.window_size,
        args.scan_mode,
        args.mutation_policy,
        args.random_seed,
        args.blosum_name,
        args.blosum_temp,
        progress=(not getattr(args, "no_progress", False)),
    )

    protein_scores = []
    for i, aa in enumerate(seq):
        for c, cname in enumerate(class_names):
            score = ablation_scores[i, c]
            # Only output rows with nonzero score
            if score != 0:
                details = ablation_details[i]
                protein_scores.append(
                    {
                        "protein_id": seq_id,
                        "residue": i + 1,  # 1-based position
                        "aa": aa,
                        "mutant_aa": details.get("mutant_aa", np.nan),
                        "class": cname,
                        "ablation_score": score,  # 0-based index
                        "Pasigned_full": details.get("Pasigned_full", np.nan),
                        "Pasigned_ablated": details.get("Pasigned_ablated", np.nan),
                        "Pclosest_full": details.get("Pclosest_full", np.nan),
                        "Pclosest_ablated": details.get("Pclosest_ablated", np.nan),
                    }
                )

    if not getattr(args, "no_progress", False):
        # Finish any in-line progress line from per-residue loop
        print("")

    return protein_scores


def compute_all_ablation_scores_parallel(sequences, model, probe, args, n_processes):
    """Compute ablation scores for all proteins in parallel using threading."""
    print(f"Using parallel processing with {n_processes} threads (shared model)")

    # Prepare data for parallel processing
    total = len(sequences)
    protein_data_list = []
    for idx, (seq_id, seq) in enumerate(sequences.items(), start=1):
        protein_data_list.append((seq_id, seq, idx, total))

    try:
        # Use threading instead of multiprocessing to share the model
        all_scores = []
        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            # Submit all tasks
            future_to_protein = {
                executor.submit(
                    process_single_protein_shared, protein_data, model, probe, args
                ): protein_data
                for protein_data in protein_data_list
            }

            # Collect results as they complete
            for future in as_completed(future_to_protein):
                protein_scores = future.result()
                all_scores.extend(protein_scores)

        return pd.DataFrame(all_scores)

    except Exception as e:
        print(f"Parallel processing failed, falling back to serial: {e}")
        return compute_all_ablation_scores_serial(sequences, model, probe, args)


def save_ablation_scores(df, args):
    """Save ablation scores to TSV file."""
    tsv_out = f"{args.output_prefix}_ablation_scores.tsv"
    df.to_csv(tsv_out, sep="\t", index=False)
    print(f"Ablation scores written to {tsv_out}")
    return tsv_out


def generate_visualizations(df, sequences, args, sdp_dict=None):
    """Generate cluster heatmaps and MSA visualizations."""
    # Get predicted classes for each protein
    protein_clusters = (
        df.groupby("protein_id")
        .apply(lambda x: x.loc[x["Pasigned_full"].idxmax(), "class"], include_groups=False)
        .reset_index()
    )
    protein_clusters.columns = ["protein_id", "predicted_class"]

    # Generate cluster heatmaps for each unique cluster
    clusters = protein_clusters["predicted_class"].unique()
    # Order clusters by size (descending) for plotting priority
    cluster_sizes = protein_clusters.groupby("predicted_class").size().sort_values(ascending=False)
    clusters_sorted = list(cluster_sizes.index)

    if args.max_clusters_to_plot is not None:
        clusters_to_plot = clusters_sorted[: args.max_clusters_to_plot]
        print(
            f"Generating plots for top {len(clusters_to_plot)} of {len(clusters_sorted)} clusters: {clusters_to_plot}"
        )
    else:
        clusters_to_plot = clusters_sorted
        print(f"Generating cluster plots for {len(clusters_to_plot)} clusters: {clusters_to_plot}")

    for cluster_name in clusters_to_plot:
        # Skip all plotting if requested
        if getattr(args, "no_plots", False):
            print(f"Skipping all plots for cluster {cluster_name} (--no-plots)")
            continue

        # Heatmap / proportional-width heatmaps
        if not getattr(args, "no_heatmap", False):
            create_cluster_heatmaps(
                df,
                protein_clusters,
                sequences,
                cluster_name,
                args.model_type,
                args.output_prefix,
                sdp_dict,
                args,
            )
        else:
            print(f"Skipping heatmap plots for cluster {cluster_name} (--no-heatmap)")

        # Alignment / colored MSA plots
        if not getattr(args, "no_alignment_plots", False):
            create_cluster_msa_with_ablation_coloring(
                df,
                protein_clusters,
                sequences,
                cluster_name,
                args.model_type,
                args.output_prefix,
                sdp_dict,
                args,
            )
        else:
            print(f"Skipping alignment/MSA plots for cluster {cluster_name} (--no-alignment-plots)")


def main():
    """Main function - orchestrates the entire ablation analysis pipeline."""
    # Parse arguments
    args = parse_arguments()

    # Load input data
    sequences = load_input_sequences(args)

    # Setup model
    model_kwargs = parse_model_arguments(args)
    device = setup_device(args)
    model = load_and_setup_model(args, model_kwargs, device)

    # Store model_kwargs in args for parallel processing
    args._model_kwargs = model_kwargs

    # Setup probe
    probe = load_and_setup_probe(args)

    # Load SDP info if provided
    sdp_dict = None
    if args.sdp_tsv:
        sdp_dict = {}
        with open(args.sdp_tsv) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1:
                    prot = parts[0]
                    coords = set()
                    for c in parts[1:]:
                        try:
                            coords.add(int(c))
                        except Exception:
                            continue
                    sdp_dict[prot] = coords

    # Compute ablation scores
    df = compute_all_ablation_scores(sequences, model, probe, args)

    # Ensure output directory exists if output_prefix includes a path
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save results
    save_ablation_scores(df, args)

    # Generate visualizations
    generate_visualizations(df, sequences, args, sdp_dict)


if __name__ == "__main__":
    main()
