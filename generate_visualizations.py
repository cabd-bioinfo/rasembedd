#!/usr/bin/env python3
"""
Improved protein embedding visualization script.

This script generates various visualizations for protein embeddings including:
- Distance heatmaps with clustering
- 2D projections (UMAP, t-SNE, PCA, PaCMAP)
- Multiple input formats support (pickle, npz, hdf5)
- Flexible output formats and customization options

Supported dimensionality reduction methods:
    - UMAP: Uniform Manifold Approximation and Projection
    - t-SNE: t-distributed Stochastic Neighbor Embedding
    - PCA: Principal Component Analysis
    - PaCMAP: Pairwise Controlled Manifold Approximation Projection (https://github.com/YingfanWang/PaCMAP)
"""

import argparse

try:
    import pacmap

    PACMAP_AVAILABLE = True
except ImportError:
    PACMAP_AVAILABLE = False
    print("Warning: PaCMAP not available. Install with: pip install pacmap")
import os
import pickle
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_random_seeds(seed: int):
    """Set random seeds for all libraries to ensure reproducibility."""
    # Set Python's built-in random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["SKLEARN_SEED"] = str(seed)

    # Set threading to single thread for deterministic results
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Set matplotlib to use Agg backend for consistency
    plt.switch_backend("Agg")

    # Try to set TensorFlow seed if available (some libraries might use it)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    # Try to set PyTorch seed if available (some libraries might use it)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Try to import h5py for HDF5 support
try:
    import h5py

    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("Warning: h5py not available. HDF5 format not supported.")


def load_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings from various file formats."""
    path = Path(embeddings_path)

    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    print(f"Loading embeddings from {embeddings_path}")

    if path.suffix == ".pkl":
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
    elif path.suffix == ".npz":
        data = np.load(embeddings_path)
        if "embeddings" in data and "ids" in data:
            embeddings = dict(zip(data["ids"], data["embeddings"]))
        else:
            # Fallback: assume first array is embeddings, second is ids
            arrays = list(data.keys())
            if len(arrays) >= 2:
                embeddings = dict(zip(data[arrays[1]], data[arrays[0]]))
            else:
                raise ValueError("NPZ file should contain 'embeddings' and 'ids' arrays")
    elif path.suffix in [".h5", ".hdf5"]:
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for HDF5 files. Install with: pip install h5py")

        embeddings = {}
        with h5py.File(embeddings_path, "r") as f:
            # Process keys in sorted order for deterministic results
            for seq_id in sorted(f.keys()):
                if "embedding" in f[seq_id]:
                    embeddings[seq_id] = f[seq_id]["embedding"][:]
                else:
                    # Fallback: use the data directly
                    embeddings[seq_id] = f[seq_id][:]
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Supported formats: .pkl, .npz, .h5, .hdf5"
        )

    print(f"Loaded {len(embeddings)} embeddings")

    # Ensure embeddings is an OrderedDict or dict with consistent ordering
    if not isinstance(embeddings, dict):
        embeddings = dict(embeddings)

    return embeddings


def load_metadata(table_path: str, sep: str = "auto") -> pd.DataFrame:
    """Load metadata table with automatic separator detection."""
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Metadata file not found: {table_path}")

    # Auto-detect separator
    if sep == "auto":
        with open(table_path, "r") as f:
            first_line = f.readline()
            if "\t" in first_line:
                sep = "\t"
            elif "," in first_line:
                sep = ","
            else:
                sep = "\t"  # Default to tab

    df = pd.read_csv(table_path, sep=sep)
    print(f"Loaded metadata for {len(df)} sequences")
    return df


def calculate_distances(embeddings: np.ndarray, metric: str) -> np.ndarray:
    """Calculate pairwise distances between embeddings."""
    print(f"Calculating {metric} distances for {embeddings.shape[0]} embeddings")

    if metric == "cosine":
        return cosine_distances(embeddings)
    elif metric == "euclidean":
        return euclidean_distances(embeddings)
    else:
        raise ValueError("Invalid distance metric. Choose 'cosine' or 'euclidean'.")


def get_color_palette(n_colors: int, palette_name: str = "glasbey") -> List:
    """Get color palette with support for different palettes."""
    # Ensure deterministic color palette generation
    np.random.seed(42)  # Fixed seed for color palette consistency

    if palette_name == "glasbey":
        return sns.color_palette(cc.glasbey, n_colors=n_colors)
    elif palette_name == "tab10":
        return sns.color_palette("tab10", n_colors=n_colors)
    elif palette_name == "Set3":
        return sns.color_palette("Set3", n_colors=n_colors)
    elif palette_name == "husl":
        return sns.color_palette("husl", n_colors=n_colors)
    else:
        return sns.color_palette(palette_name, n_colors=n_colors)


def extract_species(ids: List[str], df: pd.DataFrame, species_column: str = "species") -> List[str]:
    """Extract species from a column, fallback to UniProt ID parsing if column missing."""
    if species_column in df.columns:
        # Create a mapping from uniprot_id to species
        if "uniprot_id" in df.columns:
            species_map = dict(zip(df["uniprot_id"], df[species_column]))
            species_list = [species_map.get(uid, "UNKNOWN") for uid in ids]
            return species_list
        else:
            # If no uniprot_id column, return the whole species column
            return df[species_column].tolist()
    else:
        # Fallback to UniProt ID parsing
        print(
            f"Warning: Species column '{species_column}' not found. Available columns: {list(df.columns)}"
        )
        print("Falling back to UniProt ID parsing.")
        species_list = []
        for uid in ids:
            if "_" in uid:
                species = uid.split("_")[-1]
            else:
                species = "UNKNOWN"
            species_list.append(species)
        return species_list


def get_marker_shapes(n_species: int) -> List[str]:
    """Get different marker shapes for species visualization."""
    # Define a variety of distinguishable marker shapes
    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
        "<",
        ">",
        "p",
        "*",
        "h",
        "H",
        "+",
        "x",
        "d",
        "|",
        "_",
    ]

    # Cycle through markers if we have more species than available markers
    if n_species <= len(markers):
        return markers[:n_species]
    else:
        # Repeat markers if needed
        repeated_markers = []
        for i in range(n_species):
            repeated_markers.append(markers[i % len(markers)])
        return repeated_markers


def plot_heatmap(
    distances: np.ndarray,
    labels: List[str],
    color_labels: List[int],
    output_file: str,
    color_column: str,
    class_names: List[str],
    figsize: Tuple[int, int] = (10, 10),
    palette_name: str = "glasbey",
    show_dendrograms: bool = True,
    cluster_method: str = "average",
) -> None:
    """Generate and save an enhanced heatmap of distances."""

    print(f"Generating heatmap with {len(class_names)} classes")

    # Set theme
    sns.set_theme(style="white")

    # Get color palette
    palette = get_color_palette(len(class_names), palette_name)
    row_colors = [palette[label] for label in color_labels]

    # Create label mapping for legend
    label_mapping = {class_name: color for class_name, color in zip(class_names, palette)}

    # Create clustermap
    g = sns.clustermap(
        distances,
        xticklabels=labels,  # Always show labels
        yticklabels=labels,  # Always show labels
        cmap="viridis_r",
        figsize=figsize,
        row_colors=[row_colors],
        col_colors=[row_colors],
        cbar_pos=None,  # Remove colorbar
        method=cluster_method,
        row_cluster=show_dendrograms,
        col_cluster=show_dendrograms,
    )

    # Adjust label font size based on number of labels
    if len(labels) <= 100:
        label_size = max(4, min(8, 200 // len(labels)))
        g.ax_heatmap.tick_params(axis="both", which="major", labelsize=label_size)
    elif len(labels) <= 200:
        # For medium datasets, use smaller font
        label_size = max(3, min(6, 100 // len(labels)))
        g.ax_heatmap.tick_params(axis="both", which="major", labelsize=label_size)
    else:
        # For large datasets, use very small font or rotate labels
        label_size = max(2, min(4, 50 // len(labels)))
        g.ax_heatmap.tick_params(axis="both", which="major", labelsize=label_size, rotation=90)

    # Create legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markersize=8,
            linestyle="",
            label=class_name,
        )
        for class_name, color in label_mapping.items()
    ]

    # Position legend appropriately
    if len(class_names) <= 10:
        ncol = 1
        bbox_anchor = (1.15, 1)
    else:
        ncol = max(1, len(class_names) // 10)
        bbox_anchor = (1.15, 0.5)

    legend = g.ax_col_dendrogram.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=bbox_anchor,
        ncol=ncol,
        frameon=False,
        fontsize=min(10, max(6, 100 // len(class_names))),
    )

    # Add title
    g.fig.suptitle(f"Distance Heatmap - {color_column}", y=0.98, fontsize=14)

    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Heatmap saved to {output_file}")


def compute_projection(
    embeddings: np.ndarray, method: str, random_seed: int = 42, **kwargs
) -> np.ndarray:
    """Compute 2D projection of embeddings using specified method."""

    print(f"Computing {method} projection")

    # Set all random seeds for reproducibility
    set_random_seeds(random_seed)

    # Initialize reducer with parameters
    if method == "UMAP":
        reducer_params = {
            "random_state": random_seed,
            "n_neighbors": min(kwargs.get("n_neighbors", 15), len(embeddings) - 1),
            "min_dist": kwargs.get("min_dist", 0.1),
            "metric": kwargs.get("umap_metric", "euclidean"),
        }
        reducer = umap.UMAP(**reducer_params)
        projection = reducer.fit_transform(embeddings)
    elif method == "TSNE":
        reducer_params = {
            "random_state": random_seed,
            "perplexity": min(kwargs.get("perplexity", 30), len(embeddings) - 1),
            "max_iter": kwargs.get("max_iter", 1000),
        }
        reducer = TSNE(**reducer_params)
        projection = reducer.fit_transform(embeddings)
    elif method == "PCA":
        reducer = PCA(n_components=2, random_state=random_seed)
        reducer_params = {"n_components": 2}
        projection = reducer.fit_transform(embeddings)
    elif method == "PaCMAP":
        if not PACMAP_AVAILABLE:
            raise ImportError("PaCMAP is not installed. Install with: pip install pacmap")
        reducer_params = {
            "n_components": 2,
            "n_neighbors": min(kwargs.get("n_neighbors", 10), len(embeddings) - 1),
            "random_state": random_seed,
        }
        reducer = pacmap.PaCMAP(
            n_components=reducer_params["n_components"],
            n_neighbors=reducer_params["n_neighbors"],
            random_state=reducer_params["random_state"],
        )
        projection = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Invalid projection method. Choose 'UMAP', 'TSNE', 'PCA', or 'PaCMAP'.")

    return projection, reducer_params


def plot_projection(
    projection: np.ndarray,
    labels: List[int],
    method: str,
    class_names: List[str],
    output_file: str,
    figsize: Tuple[int, int] = (12, 9),
    palette_name: str = "glasbey",
    random_seed: int = 42,
    protein_ids: Optional[List[str]] = None,
    reducer_params: dict = None,
    **kwargs,
) -> None:
    """Generate and save a 2D projection plot from pre-computed projection."""

    print(f"Plotting {method} projection")

    # Set random seed for reproducible plotting (especially for label selection)
    set_random_seeds(random_seed)

    # Set up colors
    palette = get_color_palette(len(class_names), palette_name)
    colors = [palette[label] for label in labels]

    # Set up species-based markers if requested
    show_species = kwargs.get("show_species", False)
    species_column = kwargs.get("species_column", "species")
    df = kwargs.get("df", None)
    if show_species and protein_ids is not None and df is not None:
        species_list = extract_species(protein_ids, df, species_column)
        unique_species = list(set(species_list))
        unique_species.sort()  # Sort for consistent ordering
        species_markers = get_marker_shapes(len(unique_species))
        species_to_marker = dict(zip(unique_species, species_markers))
        markers = [species_to_marker[species] for species in species_list]
        print(f"Found {len(unique_species)} species: {unique_species}")
    else:
        markers = None
        unique_species = []
        species_to_marker = {}

    # Create plot
    plt.figure(figsize=figsize)

    # Plot points with different markers for species if requested
    if show_species and markers is not None:
        # Plot each species separately to use different markers
        for species in unique_species:
            # Use extract_species to get species list for protein_ids
            species_ids = extract_species(protein_ids, df, species_column)
            species_indices = [i for i, s in enumerate(species_ids) if s == species]
            if species_indices:
                species_projection = projection[species_indices]
                species_colors = [colors[i] for i in species_indices]
                plt.scatter(
                    species_projection[:, 0],
                    species_projection[:, 1],
                    c=species_colors,
                    marker=species_to_marker[species],
                    s=kwargs.get("point_size", 50),
                    alpha=kwargs.get("alpha", 0.7),
                    edgecolors="black",
                    linewidth=0.1,
                    label=(
                        f"{species}" if len(unique_species) <= 10 else None
                    ),  # Only show species labels if not too many
                )
    else:
        # Standard plot with same marker for all points
        scatter = plt.scatter(
            projection[:, 0],
            projection[:, 1],
            c=colors,
            s=kwargs.get("point_size", 50),
            alpha=kwargs.get("alpha", 0.7),
            edgecolors="black",
            linewidth=0.1,
        )

    # Add protein ID labels if requested
    show_labels = kwargs.get("show_labels", False)
    max_labels = kwargs.get("max_labels", 100)
    label_fontsize = kwargs.get("label_fontsize", 8)

    if show_labels and protein_ids is not None:
        n_points = len(protein_ids)

        if n_points <= max_labels:
            # Show all labels
            for i, (x, y, pid) in enumerate(zip(projection[:, 0], projection[:, 1], protein_ids)):
                plt.annotate(
                    pid,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=label_fontsize,
                    alpha=0.8,
                )
        else:
            # Show only a subset of labels to avoid overcrowding
            print(
                f"Warning: Too many points ({n_points}) to label all. Showing {max_labels} labels."
            )

            # Use deterministic selection based on protein ID hash for reproducibility
            # This ensures the same proteins are always selected regardless of order
            id_hashes = [(i, hash(pid)) for i, pid in enumerate(protein_ids)]
            id_hashes.sort(key=lambda x: x[1])  # Sort by hash for deterministic selection
            indices = [x[0] for x in id_hashes[:max_labels]]

            for i in indices:
                x, y, pid = projection[i, 0], projection[i, 1], protein_ids[i]
                plt.annotate(
                    pid,
                    (x, y),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=label_fontsize,
                    alpha=0.8,
                )

    # Create legends
    legends_to_show = []

    # Class color legend
    class_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markersize=10,
            linestyle="",
            label=class_name,
        )
        for class_name, color in zip(class_names, palette)
    ]

    # Species marker legend (if showing species)
    if show_species and len(unique_species) > 1:
        species_handles = [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                markersize=10,
                linestyle="",
                label=species,
                markerfacecolor="gray",
            )
            for species, marker in species_to_marker.items()
        ]

        # Position legends based on number of items
        if len(class_names) <= 8 and len(unique_species) <= 8:
            # Show both legends side by side
            class_legend = plt.legend(
                handles=class_handles,
                loc="upper left",
                bbox_to_anchor=(1.05, 1),
                frameon=False,
                fontsize=9,
            )
            plt.gca().add_artist(class_legend)  # Keep first legend when adding second

            species_legend = plt.legend(
                handles=species_handles,
                loc="upper left",
                bbox_to_anchor=(1.05, 0.6),
                frameon=False,
                fontsize=9,
            )
        else:
            # Show only class legend if too many items
            print("Warning: Too many classes/species for dual legend. Showing class legend only.")
            class_legend = plt.legend(
                handles=class_handles,
                loc="best",
                frameon=False,
                fontsize=min(9, max(6, 80 // len(class_names))),
            )
    else:
        # Show only class legend
        if len(class_names) <= 15:
            class_legend = plt.legend(handles=class_handles, loc="best", frameon=False, fontsize=9)
        else:
            class_legend = plt.legend(
                handles=class_handles,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                frameon=False,
                fontsize=8,
                ncol=max(1, len(class_names) // 15),
            )

    plt.title(f"{method} Projection", fontsize=16, fontweight="bold")
    plt.xlabel(f"{method} 1", fontsize=12)
    plt.ylabel(f"{method} 2", fontsize=12)

    # Add method-specific information to plot
    if method == "UMAP" and reducer_params:
        info_text = f"n_neighbors={reducer_params.get('n_neighbors', 'N/A')}, min_dist={reducer_params.get('min_dist', 'N/A')}"
    elif method == "TSNE" and reducer_params:
        info_text = f"perplexity={reducer_params.get('perplexity', 'N/A')}, max_iter={reducer_params.get('max_iter', 'N/A')}"
    elif method == "PaCMAP" and reducer_params:
        info_text = f"n_neighbors={reducer_params.get('n_neighbors', 'N/A')}"
    else:
        info_text = ""

    if info_text:
        plt.figtext(0.02, 0.02, info_text, fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"{method} projection saved to {output_file}")


def validate_inputs(
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    color_column: str,
    id_column: str = "uniprot_id",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Validate and align inputs."""

    # Check if ID column exists
    if id_column not in df.columns:
        raise ValueError(
            f"ID column '{id_column}' not found in metadata. Available columns: {list(df.columns)}"
        )

    # Check if color column exists
    if color_column not in df.columns:
        raise ValueError(
            f"Color column '{color_column}' not found in metadata. Available columns: {list(df.columns)}"
        )

    # Find common IDs
    df_ids = set(df[id_column])
    emb_ids = set(embeddings.keys())
    common_ids = df_ids.intersection(emb_ids)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between metadata and embeddings")

    # Sort common IDs for deterministic order
    common_ids = sorted(list(common_ids))

    missing_in_df = emb_ids - df_ids
    missing_in_emb = df_ids - emb_ids

    if missing_in_df:
        print(f"Warning: {len(missing_in_df)} embeddings have no metadata")
    if missing_in_emb:
        print(f"Warning: {len(missing_in_emb)} metadata entries have no embeddings")

    # Filter to common IDs
    df_filtered = df[df[id_column].isin(common_ids)].copy()

    # Ensure order consistency with sorted common_ids
    df_filtered = df_filtered.set_index(id_column).loc[common_ids].reset_index()
    embeddings_array = np.array([embeddings[uid] for uid in df_filtered[id_column]])

    print(f"Using {len(df_filtered)} sequences for visualization")

    return df_filtered, embeddings_array


def get_output_filename(
    base_name: str,
    method: str,
    color_column: str,
    embeddings_file: str,
    output_format: str = "pdf",
) -> str:
    """Generate descriptive output filename."""
    emb_name = Path(embeddings_file).stem

    # Use base_name as the output_prefix (directory or filename prefix) as provided
    output_prefix = base_name

    # Build filename components
    if "heatmap" in base_name.lower():
        # For heatmaps: prefix_heatmap_metric_color_embedding.format
        parts = base_name.split("_")
        distance_metric = "cosine"  # default
        for part in parts:
            if part.lower() in ["cosine", "euclidean"]:
                distance_metric = part.lower()
                break
        method_part = f"heatmap_{distance_metric}"
        filename_core = f"{method_part}_{color_column}_{emb_name}.{output_format}"
    else:
        # For projections: avoid repeating method/projection if already in prefix
        method_lower = method.lower()
        prefix_lower = output_prefix.lower() if output_prefix else ""
        # Remove redundant method/projection in prefix
        if (method_lower in prefix_lower) or ("projection" in prefix_lower):
            filename_core = f"{color_column}_{emb_name}.{output_format}"
        else:
            filename_core = f"{method}_{color_column}_{emb_name}.{output_format}"

    # Compose the final filename
    if output_prefix:
        # If output_prefix ends with / or is a directory, treat as directory
        if output_prefix.endswith("/") or os.path.isdir(output_prefix):
            prefix_dir = output_prefix.rstrip("/")
            # Ensure the directory exists
            os.makedirs(prefix_dir, exist_ok=True)
            filename = f"{prefix_dir}/{filename_core}"
        else:
            # If output_prefix is a path with directories, ensure parent directory exists
            parent_dir = os.path.dirname(output_prefix)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            # Insert underscore if prefix does not end with _
            if output_prefix.endswith("_"):
                filename = f"{output_prefix}{filename_core}"
            else:
                filename = f"{output_prefix}_{filename_core}"
    else:
        filename = filename_core

    return filename


def main():
    parser = argparse.ArgumentParser(
        description="Generate improved visualizations for protein embeddings. Supports UMAP, t-SNE, PCA, and PaCMAP for dimensionality reduction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_visualizations_improved.py data/metadata.tsv -e embeddings.pkl

  # Multiple color columns with single embedding file
  python generate_visualizations_improved.py data/metadata.tsv -e embeddings.pkl \
    --color_columns Family.name New Classic --methods UMAP TSNE PaCMAP

  # Multiple embedding files with single color column
  python generate_visualizations_improved.py data/metadata.tsv -e embeddings1.pkl \
    --embedding_files embeddings1.pkl embeddings2.pkl embeddings3.pkl \
    --color_column Family.name --methods UMAP PaCMAP

  # Multiple embedding files AND multiple color columns
  python generate_visualizations_improved.py data/metadata.tsv -e embeddings1.pkl \
    --embedding_files embeddings1.pkl embeddings2.pkl \
    --color_columns Family.name New --methods UMAP TSNE PCA PaCMAP

  # Custom parameters with protein ID labels and species markers
  python generate_visualizations_improved.py data/metadata.tsv -e embeddings.h5 \
    --color_column Family.name --projection_method PaCMAP --palette_name husl \
    --figsize 15 12 --output_format png --show_labels --show_species

  # Large dataset comparison across models and classifications
  python generate_visualizations_improved.py data/metadata.tsv -e model1.pkl \
    --embedding_files model1.pkl model2.pkl model3.pkl \
    --color_columns Family.name Classic New \
    --methods UMAP PaCMAP --skip_heatmap --max_labels 50
        """,
    )

    # Required arguments
    parser.add_argument("input_table", help="Path to metadata table (TSV/CSV format)")
    parser.add_argument(
        "-e", "--embeddings", help="Path to embeddings file (.pkl, .npz, .h5, .hdf5)"
    )

    # Visualization options
    parser.add_argument(
        "-c",
        "--color_column",
        default="Family.name",
        help="Column name for coloring (default: Family.name)",
    )
    parser.add_argument(
        "--color_columns",
        nargs="+",
        help="Generate visualizations for multiple color columns",
    )
    parser.add_argument(
        "--id_column",
        default="uniprot_id",
        help="Column name for sequence IDs (default: uniprot_id)",
    )
    parser.add_argument(
        "-m",
        "--distance_metric",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Distance metric for heatmap (default: cosine)",
    )
    parser.add_argument(
        "-p",
        "--projection_method",
        default="UMAP",
        choices=["UMAP", "TSNE", "PCA", "PaCMAP"],
        help="Projection method (default: UMAP)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["UMAP", "TSNE", "PCA", "PaCMAP"],
        help="Generate multiple projection methods",
    )
    parser.add_argument(
        "--embedding_files",
        nargs="+",
        help="Generate visualizations for multiple embedding files",
    )

    # Appearance options
    parser.add_argument(
        "--palette_name",
        default="glasbey",
        choices=["glasbey", "tab10", "Set3", "husl"],
        help="Color palette (default: glasbey)",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[11.69, 8.27],
        help="Figure size width height in inches (default: 11.69 8.27, A4 landscape)",
    )
    parser.add_argument(
        "--output_format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Output format (default: pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for raster formats (default: 300)",
    )

    # Label options
    parser.add_argument(
        "--show_labels",
        action="store_true",
        default=True,
        help="Show protein ID labels on projection plots (default: True)",
    )
    parser.add_argument(
        "--max_labels",
        type=int,
        default=100,
        help="Maximum number of labels to show (default: 100)",
    )
    parser.add_argument(
        "--label_fontsize",
        type=int,
        default=8,
        help="Font size for protein ID labels (default: 8)",
    )
    parser.add_argument(
        "--show_species",
        action="store_true",
        default=True,
        help="Show species as different marker shapes (from species column or UniProt IDs) (default: True)",
    )
    parser.add_argument(
        "--species_column",
        default="species",
        help="Column name for species (default: species)",
    )

    # Control options
    parser.add_argument("--skip_heatmap", action="store_true", help="Skip heatmap generation")
    parser.add_argument("--skip_projection", action="store_true", help="Skip projection generation")
    parser.add_argument("--output_prefix", default="", help="Prefix for output filenames")

    # Advanced parameters
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=15,
        help="n_neighbors parameter for UMAP and PaCMAP (default: 15 for UMAP, 10 for PaCMAP)",
    )
    parser.add_argument(
        "--min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter (default: 30)",
    )

    args = parser.parse_args()

    # Validate that either -e or --embedding_files is provided
    if not args.embeddings and not args.embedding_files:
        parser.error("Either -e/--embeddings or --embedding_files must be provided")

    try:
        # Set global random seeds for reproducibility
        set_random_seeds(args.random_seed)

        # Determine which embedding files to process
        if args.embedding_files:
            embedding_files = args.embedding_files
        elif args.embeddings:
            embedding_files = [args.embeddings]
        else:
            raise ValueError(
                "No embedding files specified"
            )  # This shouldn't happen due to validation above

        # Determine which color columns to use
        color_columns = args.color_columns if args.color_columns else [args.color_column]

        # Load metadata once
        df = load_metadata(args.input_table)

        # Validate that all color columns exist
        for color_col in color_columns:
            if color_col not in df.columns:
                raise ValueError(
                    f"Color column '{color_col}' not found in metadata. Available columns: {list(df.columns)}"
                )

        print(
            f"Processing {len(embedding_files)} embedding file(s) with {len(color_columns)} color column(s)"
        )

        # Process each embedding file
        for emb_idx, embeddings_file in enumerate(embedding_files):
            print(f"\n{'='*60}")
            print(
                f"Processing embedding file {emb_idx+1}/{len(embedding_files)}: {embeddings_file}"
            )
            print(f"{'='*60}")

            # Reset seeds for each embedding file to ensure reproducibility
            set_random_seeds(args.random_seed)

            # Load embeddings for this file
            embeddings = load_embeddings(embeddings_file)

            # Process each color column for this embedding file
            for color_idx, color_column in enumerate(color_columns):
                print(f"\n{'-'*40}")
                print(f"Processing color column {color_idx+1}/{len(color_columns)}: {color_column}")
                print(f"{'-'*40}")

                # Reset seeds for each color column to ensure reproducibility
                set_random_seeds(args.random_seed)

                # Validate and align data for this color column
                df_filtered, embeddings_array = validate_inputs(
                    df, embeddings, color_column, args.id_column
                )

                # Prepare color labels
                color_series = df_filtered[color_column].astype("category")
                color_labels = color_series.cat.codes.tolist()
                class_names = color_series.cat.categories.tolist()

                print(f"Found {len(class_names)} classes: {class_names}")

                # Generate heatmap
                if not args.skip_heatmap:
                    distances = calculate_distances(embeddings_array, args.distance_metric)
                    heatmap_output = get_output_filename(
                        f"{args.output_prefix}heatmap_{args.distance_metric}",
                        "heatmap",
                        color_column,
                        embeddings_file,
                        args.output_format,
                    )
                    plot_heatmap(
                        distances,
                        df_filtered[args.id_column].tolist(),
                        color_labels,
                        heatmap_output,
                        color_column,
                        class_names,
                        figsize=tuple(args.figsize),
                        palette_name=args.palette_name,
                    )

                # Generate projections
                if not args.skip_projection:
                    methods = args.methods if args.methods else [args.projection_method]

                    # Pre-compute projections to ensure reproducibility across different color columns
                    # Only compute once per embedding file and method combination
                    if color_idx == 0:  # First color column - compute projections
                        projections = {}
                        for method in methods:
                            projection, reducer_params = compute_projection(
                                embeddings_array,
                                method,
                                args.random_seed,
                                n_neighbors=args.n_neighbors,
                                min_dist=args.min_dist,
                                perplexity=args.perplexity,
                            )
                            projections[method] = (projection, reducer_params)

                    # Generate plots using pre-computed projections
                    for method in methods:
                        projection, reducer_params = projections[method]
                        projection_output = get_output_filename(
                            f"{args.output_prefix}projection_{method}",
                            method,
                            color_column,
                            embeddings_file,
                            args.output_format,
                        )
                        plot_projection(
                            projection,
                            color_labels,
                            method,
                            class_names,
                            projection_output,
                            figsize=tuple(args.figsize),
                            palette_name=args.palette_name,
                            random_seed=args.random_seed,
                            protein_ids=df_filtered[args.id_column].tolist(),
                            reducer_params=reducer_params,
                            show_labels=args.show_labels,
                            max_labels=args.max_labels,
                            label_fontsize=args.label_fontsize,
                            show_species=args.show_species,
                            species_column=args.species_column,
                            df=df_filtered,
                        )

        print("\n✅ All visualizations completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
