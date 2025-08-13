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
import os
import pickle
import subprocess
import sys
import tempfile
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


def create_cluster_heatmaps(
    df, protein_clusters, sequences, cluster_name, model_type, output_prefix, sdp_dict=None
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
    df, protein_clusters, sequences, cluster_name, model_type, output_prefix, sdp_dict=None
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

    # Create temporary FASTA file for MAFFT
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as temp_fasta:
        for protein_id, seq in cluster_sequences.items():
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
        for protein_id in cluster_sequences.keys():
            protein_data = df[(df["protein_id"] == protein_id) & (df["class"] == cluster_name)]
            if not protein_data.empty:
                scores = protein_data.set_index("residue")["ablation_score"]
                protein_ablation_scores[protein_id] = scores.to_dict()

        # Create colored MSA visualization
        create_colored_msa_plot(
            alignment, protein_ablation_scores, cluster_name, model_type, output_prefix, sdp_dict
        )

    finally:
        # Clean up temporary file
        os.unlink(temp_fasta_path)


def create_colored_msa_plot(
    alignment, protein_ablation_scores, cluster_name, model_type, output_prefix, sdp_dict=None
):
    """Create a colored MSA plot with ablation scores - rewritten for memory efficiency."""
    num_seqs = len(alignment)
    align_len = alignment.get_alignment_length()

    # For very large MSAs, skip visualization to avoid memory issues
    if num_seqs > 100 or align_len > 2000:
        print(
            f"Skipping MSA visualization for {cluster_name} - too large ({num_seqs} sequences, {align_len} positions)"
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
def ablation_scores_for_protein(seq_id, sequence, model, probe, window_size):
    # Use new standardized interface for residue and mean embeddings
    if not (hasattr(model, "get_residue_embeddings") and hasattr(model, "get_mean_embedding")):
        raise ValueError(
            "Model must implement get_residue_embeddings and get_mean_embedding methods."
        )
    residue_emb = model.get_residue_embeddings(sequence, seq_id)
    residue_seq_len, emb_dim = residue_emb.shape
    actual_seq_len = len(sequence)

    # Handle case where residue_emb might be shorter than sequence (due to special token removal)
    if residue_seq_len != actual_seq_len:
        print(
            f"WARNING: {seq_id} - sequence length {actual_seq_len} != residue embeddings length {residue_seq_len}"
        )
        # Pad residue_emb to match sequence length if needed
        if residue_seq_len < actual_seq_len:
            padding = np.zeros((actual_seq_len - residue_seq_len, emb_dim))
            residue_emb = np.vstack([residue_emb, padding])
        else:
            # Truncate if residue_emb is longer (shouldn't happen but just in case)
            residue_emb = residue_emb[:actual_seq_len]

    seq_len = actual_seq_len
    # Full mean-pooled embedding
    full_emb = model.get_mean_embedding(sequence, seq_id)
    # Get full sequence cluster probabilities
    full_emb_norm = probe.normalize_embedding(full_emb.reshape(1, -1))
    full_probs = probe.model.predict_proba(full_emb_norm)[0]
    # For each window, ablate and score
    ablation_scores = np.zeros((seq_len, len(full_probs)))
    # For storing extra info for output
    ablation_details = [{} for _ in range(seq_len)]
    most_probable_cluster = np.argmax(full_probs)
    full_pred = most_probable_cluster
    penalty = 100.0  # Large penalty for cluster change
    for start in range(seq_len):
        end = min(start + window_size, seq_len)
        mask = np.ones(seq_len, dtype=bool)
        mask[start:end] = False
        ablated_emb = mean_pool_embedding(residue_emb, mask)
        ablated_emb_norm = probe.normalize_embedding(ablated_emb.reshape(1, -1))
        ablated_probs = probe.model.predict_proba(ablated_emb_norm)[0]
        ablated_pred = np.argmax(ablated_probs)
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
            if sorted_full[0] == most_probable_cluster:
                competitor_full = sorted_full[1]
            else:
                competitor_full = sorted_full[0]
            margin_full = full_probs[most_probable_cluster] - full_probs[competitor_full]

            # Margin for ablated embedding
            sorted_ablated = np.argsort(ablated_probs)[::-1]
            if sorted_ablated[0] == most_probable_cluster:
                competitor_ablated = sorted_ablated[1]
            else:
                competitor_ablated = sorted_ablated[0]
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
    parser.add_argument("--window_size", type=int, required=True, help="Window size L for ablation")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files")
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
    """Load sequences from input file."""
    if args.input_type == "fasta":
        return load_sequences_from_fasta(args.input)
    else:
        return load_sequences_from_tsv(args.input, args.id_column, args.seq_column)


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


def compute_all_ablation_scores(sequences, model, probe, args):
    """Compute ablation scores for all proteins."""
    class_names = (
        probe.class_names
        if hasattr(probe, "class_names")
        else [str(i) for i in range(probe.model.coef_.shape[0])]
    )

    all_scores = []
    for seq_id, seq in sequences.items():
        ablation_scores, full_probs, ablation_details = ablation_scores_for_protein(
            seq_id, seq, model, probe, args.window_size
        )
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
                            "class": cname,
                            "ablation_score": score,  # 0-based index
                            "Pasigned_full": details.get("Pasigned_full", np.nan),
                            "Pasigned_ablated": details.get("Pasigned_ablated", np.nan),
                            "Pclosest_full": details.get("Pclosest_full", np.nan),
                            "Pclosest_ablated": details.get("Pclosest_ablated", np.nan),
                        }
                    )
    return pd.DataFrame(all_scores)


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
    print(f"Generating cluster heatmaps for {len(clusters)} clusters: {list(clusters)}")

    for cluster_name in clusters:
        create_cluster_heatmaps(
            df,
            protein_clusters,
            sequences,
            cluster_name,
            args.model_type,
            args.output_prefix,
            sdp_dict,
        )
        create_cluster_msa_with_ablation_coloring(
            df,
            protein_clusters,
            sequences,
            cluster_name,
            args.model_type,
            args.output_prefix,
            sdp_dict,
        )


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
