#!/usr/bin/env python3

"""
Clustering Evaluation Script for Protein Embeddings - Refactored Version

This script reads protein embeddings and metadata, performs clustering analysis,
and evaluates clustering performance against known protein classifications.
"""

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from joblib import Parallel, delayed
from scipy.stats import ttest_ind, wilcoxon
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# Try to import colorcet for better color palettes
try:
    import colorcet as cc

    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False
    print("Warning: colorcet not available. Using default matplotlib colors.")


@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis."""

    embedding_files: List[str]
    metadata_file: str
    output_dir: str = "clustering_results"
    id_column: str = "uniprot_id"
    label_column: str = "Family.name"
    methods: List[str] = None
    n_clusters: Optional[int] = None
    max_clusters: int = 15
    normalize: bool = False
    subsample: int = 0
    subsample_fraction: float = 0.8
    stratified_subsample: bool = False

    def __post_init__(self):
        if self.methods is None:
            self.methods = ["kmeans", "hierarchical"]


@dataclass
class ClusteringResult:
    """Results from clustering analysis."""

    cluster_labels: np.ndarray
    n_clusters: int
    metrics: Dict[str, float]


class DataLoader:
    """Handles loading and preprocessing of embeddings and metadata."""

    @staticmethod
    def load_embeddings(embedding_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from pickle file."""
        print(f"Loading embeddings from {embedding_path}")
        with open(embedding_path, "rb") as f:
            embeddings = pickle.load(f)

        print(f"Loaded {len(embeddings)} embeddings")
        print(f"Embedding dimension: {next(iter(embeddings.values())).shape[0]}")
        return embeddings

    @staticmethod
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

    @staticmethod
    def prepare_data(
        embeddings: Dict[str, np.ndarray], metadata: pd.DataFrame, id_column: str = "uniprot_id"
    ) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
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


class ClusteringEngine:
    """Handles clustering operations and evaluation."""

    @staticmethod
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

    @staticmethod
    def evaluate_clustering(
        cluster_labels: np.ndarray, true_labels: np.ndarray, embeddings: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate clustering performance using multiple metrics."""

        metrics = {}

        # External validation metrics (require true labels)
        metrics["adjusted_rand_score"] = adjusted_rand_score(true_labels, cluster_labels)
        metrics["normalized_mutual_info"] = normalized_mutual_info_score(
            true_labels, cluster_labels
        )
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

    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        method: str = "kmeans",
        max_clusters: int = 15,
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Find optimal number of clusters using multiple criteria."""

        print(f"Finding optimal number of clusters for {method}...")

        cluster_range = range(2, min(max_clusters + 1, len(embeddings)))
        metrics_by_k = {}

        for k in cluster_range:
            print(f"Testing k={k}")
            cluster_labels = self.perform_clustering(embeddings, method=method, n_clusters=k)
            metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)
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


class Visualizer:
    def plot_truth_table(
        self,
        true_labels: np.ndarray,
        cluster_labels: np.ndarray,
        label_names: List[str],
        output_path: str,
        title: str = "Clustering Truth Table",
    ):
        """Plot confusion matrix (truth table) between true labels and cluster assignments, sorted for maximal diagonal."""
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(true_labels, cluster_labels)
        # Use Hungarian algorithm to maximize diagonal
        row_ind, col_ind = linear_sum_assignment(-cm)
        cm_sorted = cm[row_ind][:, col_ind]
        # Use actual unique true labels present in filtered data
        unique_true_labels = np.unique(true_labels)
        sorted_true_labels = [
            label_names[i] if i < len(label_names) else f"Label {i}" for i in row_ind
        ]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_sorted, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Cluster Label (sorted)")
        plt.ylabel("True Label (sorted)")
        plt.title(title + " (sorted)")
        plt.yticks(np.arange(len(sorted_true_labels)) + 0.5, sorted_true_labels, rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()

    """Handles all plotting and visualization tasks."""

    @staticmethod
    def get_distinct_colors(n_colors: int, palette_name: str = "glasbey") -> np.ndarray:
        """Get distinct colors for clustering visualization."""
        if COLORCET_AVAILABLE and palette_name == "glasbey":
            if n_colors <= len(cc.glasbey):
                colors = np.array(cc.glasbey[:n_colors])
            else:
                colors = np.array([cc.glasbey[i % len(cc.glasbey)] for i in range(n_colors)])
        elif COLORCET_AVAILABLE and palette_name == "glasbey_bw":
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
                colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
                colors2 = plt.cm.tab20(np.linspace(0, 1, n_colors - 10))
                colors = np.vstack([colors1, colors2])
            else:
                colors = plt.cm.hsv(np.linspace(0, 1, n_colors))

        return colors

    def plot_cluster_optimization(
        self, metrics_by_k: Dict[int, Dict[str, float]], output_path: str
    ):
        """Plot metrics vs number of clusters."""

        k_values = sorted(metrics_by_k.keys())

        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))  # A4 landscape size
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
            axes[i].set_title(f"{metric_name} vs Number of Clusters")
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
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()

    def plot_significance_heatmap(
        self, significance_file: str, output_dir: str, title: str = "Embedding Significance"
    ):
        """Plot significance heatmaps."""
        if not os.path.exists(significance_file):
            print(f"Warning: {significance_file} not found. Skipping heatmap plot.")
            return

        df = pd.read_csv(significance_file, sep="\t")
        df["diff"] = df["embedding1_mean"] - df["embedding2_mean"]

        # Determine which p-value column to use
        pval_col = None
        for col in ["t_p_holm", "wilcoxon_p_holm"]:
            if col in df.columns:
                pval_col = col
                break

        if pval_col is None:
            print(f"Warning: No corrected p-value column found in {significance_file}")
            return

        # Create heatmaps for each metric and method
        for metric in df["metric"].unique():
            for method in df["method"].unique():
                sub = df[(df["metric"] == metric) & (df["method"] == method)]
                if sub.empty:
                    continue

                matrix = sub.pivot(index="embedding1", columns="embedding2", values="diff")
                pvals = sub.pivot(index="embedding1", columns="embedding2", values=pval_col)

                plt.figure(figsize=(10, 8))
                sns.heatmap(matrix, annot=pvals.round(3), fmt="", cmap="coolwarm", center=0)
                plt.title(f"{title}: {metric} ({method})")
                plt.xlabel("Embedding 2")
                plt.ylabel("Embedding 1")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()

                fname = f"embedding_significance_heatmap_{metric}_{method}.pdf"
                plt.savefig(os.path.join(output_dir, fname))
                plt.close()


class SubsamplingAnalyzer:
    """Handles subsampling analysis and statistical testing."""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.clustering_engine = ClusteringEngine()

    def run_subsample(
        self,
        run_idx: int,
        protein_ids: List[str],
        embeddings_dict: Dict[str, Dict[str, np.ndarray]],
        true_labels: np.ndarray,
    ) -> Dict[Tuple[str, str, str], float]:
        """Run a single subsampling iteration."""

        # Stratified or random subsampling
        if self.config.stratified_subsample:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.config.subsample_fraction, random_state=run_idx
            )
            indices = np.arange(len(protein_ids))
            split = next(sss.split(indices, true_labels))
            sampled_idx = sorted(split[1])  # test indices
        else:
            sampled_idx = sorted(
                random.sample(
                    range(len(protein_ids)), int(len(protein_ids) * self.config.subsample_fraction)
                )
            )

        sampled_ids = [protein_ids[i] for i in sampled_idx]
        sampled_labels = np.array([true_labels[i] for i in sampled_idx])

        results = {}
        for emb_name, embeddings in embeddings_dict.items():
            emb_matrix = np.array([embeddings[pid] for pid in sampled_ids if pid in embeddings])

            if self.config.normalize:
                scaler = StandardScaler()
                emb_matrix = scaler.fit_transform(emb_matrix)

            for method in self.config.methods:
                if self.config.n_clusters:
                    cluster_labels = self.clustering_engine.perform_clustering(
                        emb_matrix, method=method, n_clusters=self.config.n_clusters
                    )
                else:
                    best_k, _ = self.clustering_engine.find_optimal_clusters(
                        emb_matrix,
                        sampled_labels,
                        method=method,
                        max_clusters=self.config.max_clusters,
                    )
                    optimal_k = best_k["adjusted_rand"]
                    cluster_labels = self.clustering_engine.perform_clustering(
                        emb_matrix, method=method, n_clusters=optimal_k
                    )

                metrics = self.clustering_engine.evaluate_clustering(
                    cluster_labels, sampled_labels, emb_matrix
                )

                for metric_name, value in metrics.items():
                    results[(emb_name, method, metric_name)] = value

        return results

    def run_subsampling_analysis(
        self,
        protein_ids: List[str],
        embeddings_dict: Dict[str, Dict[str, np.ndarray]],
        true_labels: np.ndarray,
    ) -> pd.DataFrame:
        """Run complete subsampling analysis."""

        print(f"Running {self.config.subsample} subsampling runs...")

        subsample_results = Parallel(n_jobs=-1)(
            delayed(self.run_subsample)(i, protein_ids, embeddings_dict, true_labels)
            for i in range(self.config.subsample)
        )

        # Aggregate results
        records = []
        for run_idx, run_result in enumerate(subsample_results):
            for key, value in run_result.items():
                emb_name, method, metric_name = key
                records.append(
                    {
                        "run": run_idx,
                        "embedding": os.path.splitext(os.path.basename(emb_name))[0],
                        "method": method,
                        "metric": metric_name,
                        "value": value,
                    }
                )

        return pd.DataFrame(records)

    def generate_statistical_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical significance tests."""

        stats_records = []
        for metric_name in df["metric"].unique():
            for method in df["method"].unique():
                vals = {}
                for emb in df["embedding"].unique():
                    vals[emb] = df[
                        (df["metric"] == metric_name)
                        & (df["method"] == method)
                        & (df["embedding"] == emb)
                    ]["value"].values

                emb_names = list(vals.keys())
                for i in range(len(emb_names)):
                    for j in range(i + 1, len(emb_names)):
                        v1, v2 = vals[emb_names[i]], vals[emb_names[j]]
                        if len(v1) > 0 and len(v2) > 0:
                            t_stat, t_p = ttest_ind(v1, v2)
                            try:
                                w_stat, w_p = wilcoxon(v1, v2)
                            except Exception:
                                w_stat, w_p = None, None

                            stats_records.append(
                                {
                                    "metric": metric_name,
                                    "method": method,
                                    "embedding1": emb_names[i],
                                    "embedding2": emb_names[j],
                                    "embedding1_mean": np.mean(v1),
                                    "embedding2_mean": np.mean(v2),
                                    "t_stat": t_stat,
                                    "t_p": t_p,
                                    "wilcoxon_stat": w_stat,
                                    "wilcoxon_p": w_p,
                                }
                            )

        return pd.DataFrame(stats_records)


class ClusteringAnalyzer:
    """Main class that orchestrates the entire clustering analysis."""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.clustering_engine = ClusteringEngine()
        self.visualizer = Visualizer()
        self.subsampling_analyzer = SubsamplingAnalyzer(config)

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    def run_analysis(self):
        """Run the complete clustering analysis."""

        print("Starting clustering analysis...")

        # Load metadata
        metadata = self.data_loader.load_metadata(self.config.metadata_file)

        # Load all embeddings
        embeddings_dict = {}
        for embedding_file in self.config.embedding_files:
            embeddings_dict[embedding_file] = self.data_loader.load_embeddings(embedding_file)

        # Prepare common protein IDs and labels
        first_embedding = self.config.embedding_files[0]
        embedding_matrix, aligned_metadata, protein_ids = self.data_loader.prepare_data(
            embeddings_dict[first_embedding], metadata, self.config.id_column
        )

        # Filter valid labels
        valid_mask = aligned_metadata[self.config.label_column].notna()
        protein_ids = [pid for i, pid in enumerate(protein_ids) if valid_mask.iloc[i]]
        true_labels_str = aligned_metadata[self.config.label_column].values[valid_mask]
        unique_labels = sorted(list(set(true_labels_str)))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        true_labels = np.array([label_to_int[label] for label in true_labels_str])

        # Run subsampling analysis if requested
        if self.config.subsample > 0:
            self._run_subsampling_analysis(protein_ids, embeddings_dict, true_labels)
            return

        # Run regular clustering analysis
        self._run_regular_analysis(embeddings_dict, metadata, unique_labels, label_to_int)

    def _run_subsampling_analysis(
        self,
        protein_ids: List[str],
        embeddings_dict: Dict[str, Dict[str, np.ndarray]],
        true_labels: np.ndarray,
    ):
        """Run subsampling analysis workflow."""

        # Run subsampling
        df = self.subsampling_analyzer.run_subsampling_analysis(
            protein_ids, embeddings_dict, true_labels
        )

        # Save results
        df.to_csv(
            os.path.join(self.config.output_dir, "subsampling_metrics.tsv"), sep="\t", index=False
        )

        # Generate statistical tests
        stats_df = self.subsampling_analyzer.generate_statistical_tests(df)

        # Add multiple test correction
        if not stats_df.empty:
            # Add Holm-Bonferroni correction
            if "t_p" in stats_df.columns:
                _, t_pvals_corr, _, _ = multipletests(stats_df["t_p"].values, method="holm")
                stats_df["t_p_holm"] = t_pvals_corr

            stats_df.to_csv(
                os.path.join(self.config.output_dir, "subsampling_significance.tsv"),
                sep="\t",
                index=False,
            )

        # Generate plots
        self._generate_subsampling_plots(df)

        # Generate heatmaps if files exist
        if os.path.exists(os.path.join(self.config.output_dir, "subsampling_significance.tsv")):
            self.visualizer.plot_significance_heatmap(
                os.path.join(self.config.output_dir, "subsampling_significance.tsv"),
                self.config.output_dir,
                "Embedding Significance (mean diff, corrected p-value)",
            )

        print("Subsampling analysis complete. Results saved.")

    def _generate_subsampling_plots(self, df: pd.DataFrame):
        """Generate subsampling distribution plots."""
        for metric_name in df["metric"].unique():
            # Boxplots
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=df[df["metric"] == metric_name], x="embedding", y="value", hue="method"
            )
            plt.title(f"Subsampling Distribution: {metric_name}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.config.output_dir, f"subsampling_{metric_name}_boxplot.pdf")
            )
            plt.close()

            # Histograms for each embedding and method
            for method in df["method"].unique():
                for emb in df["embedding"].unique():
                    subset = df[
                        (df["metric"] == metric_name)
                        & (df["method"] == method)
                        & (df["embedding"] == emb)
                    ]
                    if len(subset) > 0:
                        plt.figure(figsize=(8, 5))
                        sns.histplot(subset["value"], bins=20, kde=True, color="skyblue")
                        plt.title(f"Histogram: {metric_name} | {emb} | {method}")
                        plt.xlabel(metric_name)
                        plt.ylabel("Frequency")
                        plt.tight_layout()
                        fname = f"subsampling_{metric_name}_hist_{emb}_{method}.pdf"
                        plt.savefig(os.path.join(self.config.output_dir, fname))
                        plt.close()

    def _run_regular_analysis(
        self,
        embeddings_dict: Dict[str, Dict[str, np.ndarray]],
        metadata: pd.DataFrame,
        unique_labels: List[str],
        label_to_int: Dict[str, int],
    ):
        """Run regular clustering analysis workflow."""

        embedding_results = {}

        # Process each embedding
        for embedding_file in self.config.embedding_files:
            print(f"\n{'='*60}\nProcessing: {embedding_file}\n{'='*60}")

            embeddings = embeddings_dict[embedding_file]
            embedding_matrix, aligned_metadata, protein_ids = self.data_loader.prepare_data(
                embeddings, metadata, self.config.id_column
            )

            # Filter valid labels
            valid_mask = aligned_metadata[self.config.label_column].notna()
            protein_ids = [pid for i, pid in enumerate(protein_ids) if valid_mask.iloc[i]]
            embedding_matrix = embedding_matrix[valid_mask]
            aligned_metadata = aligned_metadata[valid_mask].reset_index(drop=True)

            true_labels_str = aligned_metadata[self.config.label_column].values
            true_labels = np.array([label_to_int[label] for label in true_labels_str])

            print(f"Proteins after filtering: {len(protein_ids)}")
            print(f"Embedding dimensions: {embedding_matrix.shape}")

            # Normalize if requested
            if self.config.normalize:
                scaler = StandardScaler()
                embedding_matrix = scaler.fit_transform(embedding_matrix)

            # Run clustering for each method
            results = {}
            for method in self.config.methods:
                print(f"\nClustering with {method}...")

                if self.config.n_clusters:
                    cluster_labels = self.clustering_engine.perform_clustering(
                        embedding_matrix, method=method, n_clusters=self.config.n_clusters
                    )
                    n_clusters = self.config.n_clusters
                else:
                    best_k, optimization_results = self.clustering_engine.find_optimal_clusters(
                        embedding_matrix,
                        true_labels,
                        method=method,
                        max_clusters=self.config.max_clusters,
                    )
                    optimal_k = best_k["adjusted_rand"]
                    print(f"Optimal number of clusters: {optimal_k}")

                    cluster_labels = self.clustering_engine.perform_clustering(
                        embedding_matrix, method=method, n_clusters=optimal_k
                    )
                    n_clusters = optimal_k

                    # Save optimization plot
                    emb_name = os.path.splitext(os.path.basename(embedding_file))[0]
                    optimization_path = os.path.join(
                        self.config.output_dir, f"cluster_optimization_{emb_name}_{method}.pdf"
                    )
                    self.visualizer.plot_cluster_optimization(
                        optimization_results, optimization_path
                    )

                # Evaluate clustering
                metrics = self.clustering_engine.evaluate_clustering(
                    cluster_labels, true_labels, embedding_matrix
                )

                results[method] = ClusteringResult(
                    cluster_labels=cluster_labels, n_clusters=n_clusters, metrics=metrics
                )

                print(f"Number of clusters: {n_clusters}")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")

                # Plot truth table (confusion matrix) only in regular analysis
                emb_name = os.path.splitext(os.path.basename(embedding_file))[0]
                truth_table_path = os.path.join(
                    self.config.output_dir, f"truth_table_{emb_name}_{method}.pdf"
                )
                self.visualizer.plot_truth_table(
                    true_labels,
                    cluster_labels,
                    unique_labels,
                    truth_table_path,
                    title=f"Truth Table: {emb_name} ({method})",
                )
            # Store results
            embedding_results[embedding_file] = {
                "results": results,
                "protein_ids": protein_ids,
                "true_labels_str": true_labels_str,
                "true_labels": true_labels,
                "embedding_matrix": embedding_matrix,
            }

        # Save results and generate summary
        self._save_results(embedding_results)
        self._print_summary(embedding_results, unique_labels)

    def _save_results(self, embedding_results: Dict[str, Any]):
        """Save clustering results to files."""

        # Save per-embedding results
        for embedding_file, emb_result in embedding_results.items():
            emb_name = os.path.splitext(os.path.basename(embedding_file))[0]

            # Save cluster assignments
            assignments_df = pd.DataFrame(
                {
                    "protein_id": emb_result["protein_ids"],
                    "true_label": emb_result["true_labels_str"],
                    "true_label_numeric": emb_result["true_labels"],
                }
            )

            for method, result in emb_result["results"].items():
                assignments_df[f"{method}_cluster"] = result.cluster_labels

            assignments_df.to_csv(
                os.path.join(self.config.output_dir, f"{emb_name}_cluster_assignments.tsv"),
                sep="\t",
                index=False,
            )

            # Save metrics
            results_df = []
            for method, result in emb_result["results"].items():
                row = {"method": method, "n_clusters": result.n_clusters}
                row.update(result.metrics)
                results_df.append(row)

            results_df = pd.DataFrame(results_df)
            results_df.to_csv(
                os.path.join(self.config.output_dir, f"{emb_name}_clustering_results.tsv"),
                sep="\t",
                index=False,
            )

        # Create summary file
        summary_results = []
        for embedding_file, emb_result in embedding_results.items():
            emb_name = os.path.splitext(os.path.basename(embedding_file))[0]
            for method, result in emb_result["results"].items():
                row = {
                    "embedding": emb_name,
                    "method": method,
                    "n_clusters": result.n_clusters,
                }
                row.update(result.metrics)
                summary_results.append(row)

        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_df.to_csv(
                os.path.join(self.config.output_dir, "embedding_clustering_summary.tsv"),
                sep="\t",
                index=False,
            )

    def _print_summary(self, embedding_results: Dict[str, Any], unique_labels: List[str]):
        """Print analysis summary."""

        print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
        print(f"Results saved to: {self.config.output_dir}")
        print(f"Compared {len(self.config.embedding_files)} embedding files.")
        print(f"Found {len(unique_labels)} unique true labels: {unique_labels}")

        # Show best performing methods per embedding
        print(f"\nBest performing methods per embedding:")
        for embedding_file, emb_result in embedding_results.items():
            emb_name = os.path.splitext(os.path.basename(embedding_file))[0]
            results_data = []

            for method, result in emb_result["results"].items():
                row = {"method": method, "n_clusters": result.n_clusters}
                row.update(result.metrics)
                results_data.append(row)

            if results_data:
                results_df = pd.DataFrame(results_data)
                print(f"\n  {emb_name}:")
                for metric in ["adjusted_rand_score", "v_measure", "silhouette_score"]:
                    if metric in results_df.columns:
                        best_method = results_df.loc[results_df[metric].idxmax(), "method"]
                        best_score = results_df[metric].max()
                        print(f"    {metric}: {best_method} ({best_score:.4f})")


def parse_arguments() -> ClusteringConfig:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Evaluate protein clustering based on embeddings")
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
    parser.add_argument(
        "--subsample", type=int, default=0, help="Number of subsampling runs (0 to disable)"
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.8,
        help="Fraction of proteins to sample in each run",
    )
    parser.add_argument(
        "--stratified-subsample",
        action="store_true",
        help="Use stratified subsampling by true class label",
    )

    args = parser.parse_args()

    return ClusteringConfig(
        embedding_files=args.embedding_files,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        id_column=args.id_column,
        label_column=args.label_column,
        methods=args.methods,
        n_clusters=args.n_clusters,
        max_clusters=args.max_clusters,
        normalize=args.normalize,
        subsample=args.subsample,
        subsample_fraction=args.subsample_fraction,
        stratified_subsample=args.stratified_subsample,
    )


def main():
    """Main entry point."""
    try:
        config = parse_arguments()
        analyzer = ClusteringAnalyzer(config)
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
