#!/usr/bin/env python3

"""
Clustering Evaluation Script for Protein Embeddings

This script reads protein embeddings and metadata, performs clustering analysis,
and evaluates clustering performance against known protein classifications.
"""

import argparse
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

# Try to import colorcet for better color palettes
try:
    import colorcet as cc

    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False
    print("Warning: colorcet not available. Using default matplotlib colors.")


def get_distinct_colors(n_colors: int, palette_name: str = "glasbey") -> np.ndarray:
    """Get distinct colors for clustering visualization."""
    if COLORCET_AVAILABLE and palette_name == "glasbey":
        # Use glasbey palette which provides many distinct colors
        if n_colors <= len(cc.glasbey):
            colors = np.array(cc.glasbey[:n_colors])
        else:
            # Cycle through glasbey colors if we need more
            colors = np.array([cc.glasbey[i % len(cc.glasbey)] for i in range(n_colors)])
    elif COLORCET_AVAILABLE and palette_name == "glasbey_bw":
        # Alternative: glasbey_bw for better contrast
        palette = cc.glasbey_bw
        if n_colors <= len(palette):
            colors = np.array(palette[:n_colors])
        else:
            colors = np.array([palette[i % len(palette)] for i in range(n_colors)])
    else:
        # Fallback to combining multiple matplotlib palettes
        if n_colors <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
        elif n_colors <= 20:
            # Combine tab10 and tab20
            colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
            colors2 = plt.cm.tab20(np.linspace(0, 1, n_colors - 10))
            colors = np.vstack([colors1, colors2])
        else:
            # Use hsv for many colors
            colors = plt.cm.hsv(np.linspace(0, 1, n_colors))

    return colors


def load_embeddings(embedding_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings from pickle file."""
    print(f"Loading embeddings from {embedding_path}")
    with open(embedding_path, "rb") as f:
        embeddings = pickle.load(f)

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Embedding dimension: {next(iter(embeddings.values())).shape[0]}")
    return embeddings


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata from TSV file."""
    print(f"Loading metadata from {metadata_path}")

    # Try different separators
    for sep in ["\t", ",", ";"]:
        try:
            df = pd.read_csv(metadata_path, sep=sep)
            if len(df.columns) > 1:  # Successfully parsed multiple columns
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse metadata file: {metadata_path}")

    print(f"Loaded metadata for {len(df)} proteins")
    print(f"Columns: {list(df.columns)}")
    return df


def prepare_data(
    embeddings: Dict[str, np.ndarray], metadata: pd.DataFrame, id_column: str = "uniprot_id"
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Prepare aligned embeddings and labels for clustering."""

    # Get common protein IDs
    embedding_ids = set(embeddings.keys())
    metadata_ids = set(metadata[id_column].values)
    common_ids = embedding_ids.intersection(metadata_ids)

    print(f"Common proteins: {len(common_ids)}")
    print(f"Embeddings only: {len(embedding_ids - metadata_ids)}")
    print(f"Metadata only: {len(metadata_ids - embedding_ids)}")

    if len(common_ids) == 0:
        raise ValueError("No common proteins found between embeddings and metadata")

    # Align data
    common_ids = sorted(list(common_ids))
    embedding_matrix = np.array([embeddings[pid] for pid in common_ids])
    aligned_metadata = metadata[metadata[id_column].isin(common_ids)].copy()
    aligned_metadata = aligned_metadata.set_index(id_column).loc[common_ids].reset_index()

    return embedding_matrix, aligned_metadata, common_ids


def perform_clustering(
    embeddings: np.ndarray, method: str = "kmeans", n_clusters: int = None, **kwargs
) -> np.ndarray:
    """Perform clustering using specified method."""

    if method == "kmeans":
        if n_clusters is None:
            n_clusters = 8  # Default
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        labels = clusterer.fit_predict(embeddings)

    elif method == "hierarchical":
        if n_clusters is None:
            n_clusters = 8
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        labels = clusterer.fit_predict(embeddings)

    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = clusterer.fit_predict(embeddings)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    n_clusters_found = len(np.unique(labels))
    print(f"Clustering with {method}: found {n_clusters_found} clusters")

    return labels


def evaluate_clustering(
    cluster_labels: np.ndarray, true_labels: np.ndarray, embeddings: np.ndarray
) -> Dict[str, float]:
    """Evaluate clustering performance using multiple metrics."""

    metrics = {}

    # External validation metrics (require true labels)
    metrics["adjusted_rand_score"] = adjusted_rand_score(true_labels, cluster_labels)
    metrics["normalized_mutual_info"] = normalized_mutual_info_score(true_labels, cluster_labels)
    metrics["homogeneity"] = homogeneity_score(true_labels, cluster_labels)
    metrics["completeness"] = completeness_score(true_labels, cluster_labels)
    metrics["v_measure"] = v_measure_score(true_labels, cluster_labels)

    # Internal validation metrics (only need embeddings and cluster labels)
    if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters
        metrics["silhouette_score"] = silhouette_score(embeddings, cluster_labels)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(embeddings, cluster_labels)
        metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, cluster_labels)
    else:
        metrics["silhouette_score"] = -1.0
        metrics["calinski_harabasz_score"] = 0.0
        metrics["davies_bouldin_score"] = float("inf")

    return metrics


def plot_clustering_results(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    protein_ids: List[str],
    output_dir: str,
    method_name: str,
    true_labels_str=None,
):
    """Create visualizations of clustering results."""

    os.makedirs(output_dir, exist_ok=True)

    # Create color mapping for consistent coloring
    import matplotlib.colors as mcolors
    from scipy.optimize import linear_sum_assignment

    # Get unique labels
    unique_true = np.unique(true_labels)
    unique_clusters = np.unique(cluster_labels)

    # Create contingency matrix for optimal matching
    contingency = np.zeros((len(unique_clusters), len(unique_true)))
    for i, cluster in enumerate(unique_clusters):
        for j, true_label in enumerate(unique_true):
            contingency[i, j] = np.sum((cluster_labels == cluster) & (true_labels == true_label))

    # Find optimal matching using Hungarian algorithm
    cost_matrix = -contingency  # maximize overlap
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create color mapping using distinct colors
    max_colors_needed = max(len(unique_true), len(unique_clusters))
    colors = get_distinct_colors(max_colors_needed, "glasbey")

    # Map cluster colors to best matching true labels
    cluster_color_map = {}
    true_color_map = {}

    # Assign colors to true labels first
    for i, true_label in enumerate(unique_true):
        true_color_map[true_label] = colors[i]

    # Assign colors to clusters based on best match
    matched_clusters = set()
    for i in range(len(row_ind)):
        cluster_idx = row_ind[i]
        if i < len(col_ind):
            matched_true_idx = col_ind[i]
            cluster_color_map[unique_clusters[cluster_idx]] = colors[matched_true_idx]
            matched_clusters.add(cluster_idx)

    # Assign colors to unmatched clusters
    color_idx = len(unique_true)
    for i, cluster in enumerate(unique_clusters):
        if i not in matched_clusters:
            if color_idx < len(colors):
                cluster_color_map[cluster] = colors[color_idx]
                color_idx += 1
            else:
                # Fallback to cycling through colors
                cluster_color_map[cluster] = colors[color_idx % len(colors)]
                color_idx += 1

    # Create color arrays for plotting
    true_colors = [true_color_map[label] for label in true_labels]
    cluster_colors = [cluster_color_map[label] for label in cluster_labels]

    # Dimensionality reduction for visualization
    print("Computing dimensionality reductions for visualization...")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(embeddings)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    tsne_coords = tsne.fit_transform(embeddings)

    # UMAP
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_coords = umap_reducer.fit_transform(embeddings)

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))  # A4 landscape size in inches

    reductions = [
        (pca_coords, "PCA", f"PCA (explained variance: {pca.explained_variance_ratio_.sum():.3f})"),
        (tsne_coords, "t-SNE", "t-SNE"),
        (umap_coords, "UMAP", "UMAP"),
    ]

    for i, (coords, name, title) in enumerate(reductions):
        # Plot true labels with consistent colors (now on top)
        axes[0, i].scatter(coords[:, 0], coords[:, 1], c=true_colors, alpha=0.7, s=50)
        axes[0, i].set_title(f"{title} - True Labels")
        axes[0, i].set_xlabel(f"{name} 1")
        axes[0, i].set_ylabel(f"{name} 2")
    # Add legend for true labels at the top right of the page (figure)
    if true_labels_str is not None:
        import matplotlib.patches as mpatches

        legend_handles = []
        for true_label in unique_true:
            if true_labels_str is not None:
                label_str = None
                for j, numeric_label in enumerate(true_labels):
                    if numeric_label == true_label:
                        label_str = true_labels_str[j]
                        break
                if label_str is None:
                    label_str = str(true_label)
            else:
                label_str = str(true_label)
            patch = mpatches.Patch(color=true_color_map[true_label], label=label_str)
            legend_handles.append(patch)
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(1.15, 1),
            title="True Labels",
            fontsize="small",
            title_fontsize="small",
            ncol=1,
        )

    # Plot predicted clusters with consistent colors (now on bottom)
    for i, (coords, name, title) in enumerate(reductions):
        axes[1, i].scatter(coords[:, 0], coords[:, 1], c=cluster_colors, alpha=0.7, s=50)
        axes[1, i].set_title(f"{title} - Predicted Clusters ({method_name})")
        axes[1, i].set_xlabel(f"{name} 1")
        axes[1, i].set_ylabel(f"{name} 2")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"clustering_{method_name}.pdf"), format="pdf", bbox_inches="tight"
    )
    plt.close()

    # Confusion matrix-like comparison
    create_cluster_comparison_plot(
        cluster_labels, true_labels, output_dir, method_name, true_labels_str
    )


def create_cluster_comparison_plot(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    output_dir: str,
    method_name: str,
    true_labels_str=None,
):
    """Create a heatmap showing cluster vs true label assignments."""

    # Create contingency table
    unique_clusters = np.unique(cluster_labels)
    unique_true = np.unique(true_labels)

    # Get true label names
    if true_labels_str is not None:
        # Create mapping from numeric labels to string labels
        label_mapping = {}
        for i, label_str in enumerate(true_labels_str):
            label_mapping[true_labels[i]] = label_str
        label_names = [label_mapping[t] for t in unique_true]
    else:
        # Fallback to numeric labels
        label_names = [str(l) for l in unique_true]

    contingency = np.zeros((len(unique_clusters), len(unique_true)))
    for i, cluster in enumerate(unique_clusters):
        for j, true_label in enumerate(unique_true):
            contingency[i, j] = np.sum((cluster_labels == cluster) & (true_labels == true_label))
    # Sort rows and columns to maximize diagonal values
    from scipy.optimize import linear_sum_assignment

    cost_matrix = -contingency  # maximize diagonal
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sorted_contingency = contingency[row_ind][:, col_ind]
    sorted_cluster_labels = [f"Cluster_{unique_clusters[i]}" for i in row_ind]
    sorted_true_labels = [str(label_names[j]) for j in col_ind]
    # Create heatmap
    plt.figure(figsize=(11.7, 8.3))  # A4 landscape size in inches
    sns.heatmap(
        sorted_contingency,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=sorted_true_labels,
        yticklabels=sorted_cluster_labels,
    )
    plt.title(f"Cluster Assignment Matrix - {method_name}")
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Clusters")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"cluster_matrix_{method_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def find_optimal_clusters(
    embeddings: np.ndarray, true_labels: np.ndarray, method: str = "kmeans", max_clusters: int = 15
) -> Tuple[int, Dict]:
    """Find optimal number of clusters using multiple criteria."""

    print(f"Finding optimal number of clusters for {method}...")

    cluster_range = range(2, min(max_clusters + 1, len(embeddings)))
    metrics_by_k = {}

    for k in cluster_range:
        print(f"Testing k={k}")
        cluster_labels = perform_clustering(embeddings, method=method, n_clusters=k)
        metrics = evaluate_clustering(cluster_labels, true_labels, embeddings)
        metrics_by_k[k] = metrics

    # Find best k for different metrics
    best_k = {}
    best_k["silhouette"] = max(cluster_range, key=lambda k: metrics_by_k[k]["silhouette_score"])
    best_k["calinski_harabasz"] = max(
        cluster_range, key=lambda k: metrics_by_k[k]["calinski_harabasz_score"]
    )
    best_k["davies_bouldin"] = min(
        cluster_range, key=lambda k: metrics_by_k[k]["davies_bouldin_score"]
    )
    best_k["adjusted_rand"] = max(
        cluster_range, key=lambda k: metrics_by_k[k]["adjusted_rand_score"]
    )
    best_k["v_measure"] = max(cluster_range, key=lambda k: metrics_by_k[k]["v_measure"])

    return best_k, metrics_by_k


def plot_cluster_optimization(metrics_by_k: Dict, output_dir: str, method_name: str):
    """Plot metrics vs number of clusters."""

    k_values = sorted(metrics_by_k.keys())

    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))  # A4 landscape size in inches
    axes = axes.flatten()
    metrics_to_plot = [
        ("silhouette_score", "Silhouette Score"),
        ("calinski_harabasz_score", "Calinski-Harabasz Score"),
        ("davies_bouldin_score", "Davies-Bouldin Score"),
        ("adjusted_rand_score", "Adjusted Rand Score"),
        ("v_measure", "V-Measure Score"),
        ("normalized_mutual_info", "Normalized Mutual Information"),
    ]
    for i, (metric_key, metric_name) in enumerate(metrics_to_plot):
        values = [metrics_by_k[k][metric_key] for k in k_values]
        axes[i].plot(k_values, values, "bo-", linewidth=2, markersize=6)
        axes[i].set_xlabel("Number of Clusters")
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f"{metric_name} vs Number of Clusters ({method_name})")
        axes[i].grid(True, alpha=0.3)
        # Mark the best value
        if metric_key == "davies_bouldin_score":  # Lower is better
            best_k = k_values[np.argmin(values)]
        else:  # Higher is better
            best_k = k_values[np.argmax(values)]
        best_value = metrics_by_k[best_k][metric_key]
        axes[i].axvline(x=best_k, color="red", linestyle="--", alpha=0.7)
        axes[i].text(
            best_k,
            best_value,
            f"k={best_k}",
            verticalalignment="bottom",
            horizontalalignment="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"cluster_optimization_{method_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


def main():
    try:
        print("Starting clustering_evaluation.py...")
        parser = argparse.ArgumentParser(
            description="Evaluate protein clustering based on embeddings"
        )
        parser.add_argument(
            "embedding_files", nargs="+", help="Paths to embedding pickle files (space-separated)"
        )
        parser.add_argument("metadata_file", help="Path to metadata file (TSV/CSV)")
        parser.add_argument(
            "--output-dir", "-o", default="clustering_results", help="Output directory for results"
        )
        parser.add_argument(
            "--id-column", default="uniprot_id", help="Column name for protein IDs in metadata"
        )
        parser.add_argument(
            "--label-column", default="Family.name", help="Column name for true labels in metadata"
        )
        parser.add_argument(
            "--methods",
            nargs="+",
            default=["kmeans", "hierarchical"],
            choices=["kmeans", "hierarchical", "dbscan"],
            help="Clustering methods to use",
        )
        parser.add_argument(
            "--n-clusters", type=int, help="Number of clusters (if not specified, will optimize)"
        )
        parser.add_argument(
            "--max-clusters",
            type=int,
            default=15,
            help="Maximum number of clusters to test during optimization",
        )
        parser.add_argument(
            "--normalize", action="store_true", help="Normalize embeddings before clustering"
        )
        args = parser.parse_args()
        print(f"Embedding files: {args.embedding_files}")
        print(f"Metadata file: {args.metadata_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"ID column: {args.id_column}")
        print(f"Label column: {args.label_column}")
        print(f"Clustering methods: {args.methods}")
        print(f"n_clusters: {args.n_clusters}")
        print(f"max_clusters: {args.max_clusters}")
        print(f"Normalize: {args.normalize}")

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Load metadata
        metadata = load_metadata(args.metadata_file)
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback

        traceback.print_exc()
        return

    # Store clustering results for each embedding
    embedding_results = {}
    common_protein_ids = None
    true_labels_str = None
    true_labels = None
    label_to_int = None
    unique_labels = None
    aligned_metadata = None

    # For each embedding file
    for embedding_file in args.embedding_files:
        print(f"\n{'='*60}\nProcessing embedding file: {embedding_file}\n{'='*60}")
        embeddings = load_embeddings(embedding_file)
        embedding_matrix, aligned_metadata_tmp, protein_ids = prepare_data(
            embeddings, metadata, args.id_column
        )
        # Get true labels (only once, for common proteins)
        if common_protein_ids is None:
            if args.label_column not in aligned_metadata_tmp.columns:
                print(f"Available columns: {list(aligned_metadata_tmp.columns)}")
                raise ValueError(f"Label column '{args.label_column}' not found in metadata")
            # Filter out rows with NaN in the label column
            valid_mask = aligned_metadata_tmp[args.label_column].notna()
            if not np.all(valid_mask):
                print(
                    f"Warning: {np.sum(~valid_mask)} proteins have missing labels and will be excluded."
                )
            embedding_matrix = embedding_matrix[valid_mask]
            aligned_metadata_tmp = aligned_metadata_tmp[valid_mask]
            protein_ids = [pid for i, pid in enumerate(protein_ids) if valid_mask.iloc[i]]
            true_labels_str = aligned_metadata_tmp[args.label_column].values
            unique_labels = sorted(list(set(true_labels_str)))
            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            true_labels = np.array([label_to_int[label] for label in true_labels_str])
            common_protein_ids = protein_ids
            aligned_metadata = aligned_metadata_tmp
        else:
            # Align to common_protein_ids
            embedding_matrix = np.array(
                [embeddings[pid] for pid in common_protein_ids if pid in embeddings]
            )
            # If any protein missing, warn and skip
            if embedding_matrix.shape[0] != len(common_protein_ids):
                print(
                    f"Warning: {len(common_protein_ids) - embedding_matrix.shape[0]} proteins missing in {embedding_file}, skipping."
                )
                continue
        # Normalize embeddings if requested
        if args.normalize:
            print("Normalizing embeddings...")
            scaler = StandardScaler()
            embedding_matrix = scaler.fit_transform(embedding_matrix)
        # Results storage for this embedding
        all_results = {}
        # For each clustering method
        for method in args.methods:
            print(f"\n{'-'*40}\nEvaluating {method} clustering\n{'-'*40}")
            if args.n_clusters:
                cluster_labels = perform_clustering(
                    embedding_matrix, method=method, n_clusters=args.n_clusters
                )
                metrics = evaluate_clustering(cluster_labels, true_labels, embedding_matrix)
                all_results[method] = {
                    "n_clusters": args.n_clusters,
                    "metrics": metrics,
                    "cluster_labels": cluster_labels,
                }
            else:
                best_k, metrics_by_k = find_optimal_clusters(
                    embedding_matrix, true_labels, method=method, max_clusters=args.max_clusters
                )
                optimal_k = best_k["adjusted_rand"]
                cluster_labels = perform_clustering(
                    embedding_matrix, method=method, n_clusters=optimal_k
                )
                metrics = evaluate_clustering(cluster_labels, true_labels, embedding_matrix)

                # Generate optimization plot with embedding-specific name
                emb_base = os.path.splitext(os.path.basename(embedding_file))[0]
                plot_cluster_optimization(metrics_by_k, args.output_dir, f"{emb_base}_{method}")

                all_results[method] = {
                    "n_clusters": optimal_k,
                    "metrics": metrics,
                    "cluster_labels": cluster_labels,
                    "optimization": metrics_by_k,
                    "best_k": best_k,
                }
            # Generate and save plots for each embedding and method
            emb_base = os.path.splitext(os.path.basename(embedding_file))[0]
            plot_dir = os.path.join(args.output_dir)
            plot_clustering_results(
                embedding_matrix,
                cluster_labels,
                true_labels,
                common_protein_ids,
                plot_dir,
                f"{emb_base}_{method}",
                true_labels_str,
            )
        embedding_results[embedding_file] = {
            "results": all_results,
            "protein_ids": common_protein_ids,
            "true_labels_str": true_labels_str,
            "true_labels": true_labels,
            "unique_labels": unique_labels,
            "label_to_int": label_to_int,
            "aligned_metadata": aligned_metadata,
        }
    # Compare cluster assignments between embeddings
    print(f"\n{'='*60}\nComparing cluster assignments between embeddings\n{'='*60}")
    comparison_rows = []
    embedding_files = list(embedding_results.keys())
    for i in range(len(embedding_files)):
        for j in range(i + 1, len(embedding_files)):
            emb1 = embedding_files[i]
            emb2 = embedding_files[j]
            for method in args.methods:
                if (
                    method in embedding_results[emb1]["results"]
                    and method in embedding_results[emb2]["results"]
                ):
                    clust1 = embedding_results[emb1]["results"][method]["cluster_labels"]
                    clust2 = embedding_results[emb2]["results"][method]["cluster_labels"]
                    ari = adjusted_rand_score(clust1, clust2)
                    nmi = normalized_mutual_info_score(clust1, clust2)
                    comparison_rows.append(
                        {
                            "embedding_1": os.path.basename(emb1),
                            "embedding_2": os.path.basename(emb2),
                            "method": method,
                            "adjusted_rand_index": ari,
                            "normalized_mutual_info": nmi,
                        }
                    )
                    print(
                        f"{method}: {os.path.basename(emb1)} vs {os.path.basename(emb2)} -> ARI={ari:.4f}, NMI={nmi:.4f}"
                    )
    # Save comparison results
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(
            os.path.join(args.output_dir, "embedding_clustering_comparison.tsv"),
            sep="\t",
            index=False,
        )
        print(
            f"Clustering comparison results saved to: {os.path.join(args.output_dir, 'embedding_clustering_comparison.tsv')}"
        )
    else:
        print("No clustering comparison results to save.")
    # Save per-embedding results as before
    for embedding_file, emb_result in embedding_results.items():
        all_results = emb_result["results"]
        protein_ids = emb_result["protein_ids"]
        true_labels_str = emb_result["true_labels_str"]
        true_labels = emb_result["true_labels"]
        assignments_df = pd.DataFrame(
            {
                "protein_id": protein_ids,
                "true_label": true_labels_str,
                "true_label_numeric": true_labels,
            }
        )
        for method, result in all_results.items():
            assignments_df[f"{method}_cluster"] = result["cluster_labels"]
        out_name = (
            os.path.splitext(os.path.basename(embedding_file))[0] + "_cluster_assignments.tsv"
        )
        assignments_df.to_csv(os.path.join(args.output_dir, out_name), sep="\t", index=False)
        # Save metrics summary
        results_df = []
        for method, result in all_results.items():
            row = {"method": method, "n_clusters": result["n_clusters"]}
            row.update(result["metrics"])
            results_df.append(row)
        results_df = pd.DataFrame(results_df)
        out_name2 = (
            os.path.splitext(os.path.basename(embedding_file))[0] + "_clustering_results.tsv"
        )
        results_df.to_csv(os.path.join(args.output_dir, out_name2), sep="\t", index=False)

    # Create summary file that concatenates all clustering results with embedding column
    summary_results = []
    for embedding_file, emb_result in embedding_results.items():
        embedding_name = os.path.splitext(os.path.basename(embedding_file))[0]
        all_results = emb_result["results"]
        for method, result in all_results.items():
            row = {
                "embedding": embedding_name,
                "method": method,
                "n_clusters": result["n_clusters"],
            }
            row.update(result["metrics"])
            summary_results.append(row)

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv(
            os.path.join(args.output_dir, "embedding_clustering_summary.tsv"), sep="\t", index=False
        )
        print(
            f"Summary results saved to: {os.path.join(args.output_dir, 'embedding_clustering_summary.tsv')}"
        )

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Compared {len(embedding_files)} embedding files.")
    if common_protein_ids:
        print(f"Common proteins across all embeddings: {len(common_protein_ids)}")
        print(f"True classes: {len(unique_labels)}")
        print(f"Found {len(unique_labels)} unique true labels: {unique_labels}")

    # Show best performing methods per embedding
    print(f"\nBest performing methods per embedding:")
    for embedding_file, emb_result in embedding_results.items():
        embedding_name = os.path.splitext(os.path.basename(embedding_file))[0]
        all_results = emb_result["results"]
        if all_results:
            results_df = []
            for method, result in all_results.items():
                row = {"method": method, "n_clusters": result["n_clusters"]}
                row.update(result["metrics"])
                results_df.append(row)
            results_df = pd.DataFrame(results_df)

            print(f"\n  {embedding_name}:")
            for metric in ["adjusted_rand_score", "v_measure", "silhouette_score"]:
                best_method = results_df.loc[results_df[metric].idxmax(), "method"]
                best_score = results_df[metric].max()
                print(f"    {metric}: {best_method} ({best_score:.4f})")


if __name__ == "__main__":
    main()
