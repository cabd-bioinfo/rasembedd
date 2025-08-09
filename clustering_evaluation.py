#!/usr/bin/env python3

"""
Clustering Evaluation Script for Protein Embeddings - Refactored Version

This script reads protein embeddings and metadata, performs clustering analysis,
and evaluates clustering performance against known protein classifications.
"""

import argparse
import os
import pickle
import warnings

# Suppress specific deprecation warning from HDBSCAN library
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import ttest_ind, wilcoxon
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from statsmodels.stats.multitest import multipletests

# Try to import colorcet for better color palettes
try:
    import colorcet as cc

    COLORCET_AVAILABLE = True
except ImportError:
    COLORCET_AVAILABLE = False
    print("Warning: colorcet not available. Using default matplotlib colors.")


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


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
    k_selection_metric: str = "silhouette"
    normalization_method: str = "l2"
    # Normalization pipeline options (optional; used when normalization_method == 'pipeline' or any is set)
    norm_center: bool = False
    norm_scale: bool = False
    # PCA components: int for component count, float in (0,1] for variance retained; 0 disables PCA
    norm_pca_components: float | int = 0
    # Apply L2 at the end of pipeline (only when pipeline is active)
    norm_l2: bool = True
    clustering_options: Dict[str, Any] = None
    subsample: int = 0
    subsample_fraction: float = 0.8
    stratified_subsample: bool = False

    def __post_init__(self):
        if self.methods is None:
            self.methods = ["kmeans", "hierarchical"]
        if self.clustering_options is None:
            self.clustering_options = {}


@dataclass
class ClusteringResult:
    """Results from clustering analysis."""

    cluster_labels: np.ndarray
    n_clusters: int
    metrics: Dict[str, float]
    params: Dict[str, Any]


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

        # Validate that the ID column exists in metadata
        if id_column not in metadata.columns:
            available_columns = ", ".join(metadata.columns.tolist())
            raise ValueError(
                f"ID column '{id_column}' not found in metadata. "
                f"Available columns: {available_columns}"
            )

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


class EmbeddingNormalizer:
    """Handles different normalization methods for embeddings."""

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray, method: str = "standard") -> np.ndarray:
        """Apply normalization to embeddings using the specified method.

        Args:
            embeddings: Input embeddings of shape (n_samples, n_features)
            method: Normalization method ("standard", "l2", "pca", "zca", "none")

        Returns:
            Normalized embeddings of the same shape
        """
        if method == "none" or not method:
            return embeddings

        if method == "standard":
            return EmbeddingNormalizer._standard_normalization(embeddings)
        elif method == "l2":
            return EmbeddingNormalizer._l2_normalization(embeddings)
        elif method == "pca":
            return EmbeddingNormalizer._pca_whitening(embeddings)
        elif method == "zca":
            return EmbeddingNormalizer._zca_whitening(embeddings)
        else:
            available_methods = ["standard", "l2", "pca", "zca", "none"]
            raise ValueError(
                f"Unknown normalization method '{method}'. Available methods: {available_methods}"
            )

    @staticmethod
    def _standard_normalization(embeddings: np.ndarray) -> np.ndarray:
        """Standard normalization (z-score): mean=0, std=1 for each feature."""
        scaler = StandardScaler()
        return scaler.fit_transform(embeddings)

    @staticmethod
    def _l2_normalization(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalization: each sample has unit norm."""
        return normalize(embeddings, norm="l2", axis=1)

    @staticmethod
    def _pca_whitening(embeddings: np.ndarray) -> np.ndarray:
        """PCA whitening: decorrelate features and scale to unit variance."""
        # Center the data
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean

        # Compute PCA
        pca = PCA(whiten=True)
        whitened = pca.fit_transform(centered)

        return whitened

    @staticmethod
    def _zca_whitening(embeddings: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """ZCA whitening: decorrelate features while preserving original space structure."""
        # Center the data
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Compute eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Add small epsilon to prevent division by zero
        eigenvalues = eigenvalues + epsilon

        # Compute ZCA whitening matrix
        # ZCA = V * D^(-1/2) * V^T where V is eigenvectors, D is eigenvalues
        sqrt_inv_eigenvalues = np.diag(1.0 / np.sqrt(eigenvalues))
        whitening_matrix = eigenvectors @ sqrt_inv_eigenvalues @ eigenvectors.T

        # Apply whitening
        whitened = centered @ whitening_matrix.T

        return whitened

    @staticmethod
    def normalize_pipeline(
        embeddings: np.ndarray,
        *,
        center: bool = False,
        scale: bool = False,
        pca_components: float | int = 0,
        l2: bool = True,
        random_state: int = 42,
    ) -> tuple[np.ndarray, dict]:
        """Three-step normalization pipeline.

        1) Standard normalization with optional centering and scaling
        2) PCA dimensionality reduction (components=int or variance in (0,1]); 0 disables PCA
        3) Optional L2 per-sample normalization

        Returns:
            tuple: (normalized_embeddings, info_dict)
            info_dict contains PCA information like n_components_ and explained_variance_ratio_
        """
        X = embeddings
        info = {}

        # Step 1: Standardization
        if center or scale:
            scaler = StandardScaler(with_mean=center, with_std=scale)
            X = scaler.fit_transform(X)

        # Step 2: PCA reduction
        if isinstance(pca_components, (int, float)) and pca_components != 0:
            # Interpret pca_components as follows:
            # - int (>=1): number of components
            # - float > 1: treat as integer component count (e.g., 256.0 -> 256)
            # - 0 < float < 1: fraction of variance to retain
            # - float == 1.0: keep all components (use None to avoid invalid 1.0)
            if isinstance(pca_components, int):
                n_components: int | float | None = max(int(pca_components), 1)
            else:  # float
                if pca_components > 1.0:
                    n_components = max(int(round(pca_components)), 1)
                elif 0.0 < pca_components < 1.0:
                    n_components = pca_components
                else:  # pca_components == 1.0 (keep all components)
                    n_components = None

            # Cap integer components at maximum possible value to avoid errors
            if isinstance(n_components, int):
                max_components = min(X.shape[0] - 1, X.shape[1])  # -1 for degrees of freedom
                if n_components > max_components:
                    print(
                        f"Note: Reducing PCA components from {n_components} to {max_components} (max possible with {X.shape[0]} samples)"
                    )
                    n_components = max_components

            pca = PCA(n_components=n_components, random_state=random_state)
            X = pca.fit_transform(X)

            # Store PCA information
            info["pca_n_components"] = pca.n_components_
            info["pca_explained_variance_ratio_sum"] = pca.explained_variance_ratio_.sum()
            info["pca_requested_components"] = pca_components

        # Step 3: L2 normalization
        if l2:
            X = normalize(X, norm="l2", axis=1)

        return X, info


class ClusteringEngine:
    """Handles clustering operations and evaluation."""

    @staticmethod
    def perform_clustering(
        embeddings: np.ndarray, method: str = "kmeans", n_clusters: int = None, **kwargs
    ) -> np.ndarray:
        """Perform clustering using specified method."""

        # Filter out auto_params information before passing to clustering algorithms
        clustering_kwargs = {k: v for k, v in kwargs.items() if k != "_auto_params"}

        if method == "kmeans":
            if n_clusters is None:
                n_clusters = 8  # Default
            # Set default random_state if not provided in clustering_kwargs
            if "random_state" not in clustering_kwargs:
                clustering_kwargs["random_state"] = 42
            clusterer = KMeans(n_clusters=n_clusters, **clustering_kwargs)
            labels = clusterer.fit_predict(embeddings)

        elif method == "hierarchical":
            if n_clusters is None:
                n_clusters = 8
            # Validate ward linkage with euclidean metric
            linkage = clustering_kwargs.get("linkage", "complete")
            metric = clustering_kwargs.get("metric", "euclidean")
            if linkage == "ward" and metric != "euclidean":
                raise ValueError("Ward linkage requires euclidean metric")
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, **clustering_kwargs)
            labels = clusterer.fit_predict(embeddings)

        elif method == "dbscan":
            # Extract eps and min_samples, remove from clustering_kwargs to avoid duplication
            eps = clustering_kwargs.pop("eps", 0.5)
            min_samples = clustering_kwargs.pop("min_samples", 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, **clustering_kwargs)
            labels = clusterer.fit_predict(embeddings)

        elif method == "hdbscan":
            # Extract min_cluster_size and min_samples, remove from clustering_kwargs to avoid duplication
            min_cluster_size = clustering_kwargs.pop("min_cluster_size", 5)
            min_samples = clustering_kwargs.pop("min_samples", None)  # HDBSCAN default is None
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size, min_samples=min_samples, **clustering_kwargs
            )
            labels = clusterer.fit_predict(embeddings)

        elif method == "spectral":
            # Spectral clustering requires n_clusters
            if n_clusters is None:
                n_clusters = 8
            # Provide a default random_state for reproducibility unless explicitly set
            if "random_state" not in clustering_kwargs:
                clustering_kwargs["random_state"] = 42
            clusterer = SpectralClustering(n_clusters=n_clusters, **clustering_kwargs)
            labels = clusterer.fit_predict(embeddings)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        n_clusters_found = len(np.unique(labels))
        print(f"Clustering with {method}: found {n_clusters_found} clusters")

        return labels

    @staticmethod
    def _calculate_inertia(embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia) for elbow method."""
        inertia = 0.0
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
            cluster_points = embeddings[cluster_labels == cluster_id]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia

    @staticmethod
    def _calculate_wcss(embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares for hierarchical clustering."""
        return ClusteringEngine._calculate_inertia(embeddings, cluster_labels)

    @staticmethod
    def _find_elbow_point(k_values: List[int], inertia_values: List[float]) -> int:
        """Find elbow point using the kneedle algorithm approximation."""
        if len(k_values) < 3:
            return k_values[0] if k_values else 2

        # Normalize the data
        k_norm = np.array(k_values, dtype=float)
        inertia_norm = np.array(inertia_values, dtype=float)

        # Normalize to [0,1] range
        k_norm = (k_norm - k_norm.min()) / (k_norm.max() - k_norm.min())
        inertia_norm = (inertia_norm - inertia_norm.min()) / (
            inertia_norm.max() - inertia_norm.min()
        )

        # Calculate the distance from each point to the line connecting first and last points
        distances = []
        for i in range(len(k_norm)):
            # Line from first to last point: y = mx + b
            x1, y1 = k_norm[0], inertia_norm[0]
            x2, y2 = k_norm[-1], inertia_norm[-1]

            # Distance from point to line
            xi, yi = k_norm[i], inertia_norm[i]
            if x2 != x1:  # Avoid division by zero
                distance = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1) / np.sqrt(
                    (y2 - y1) ** 2 + (x2 - x1) ** 2
                )
            else:
                distance = abs(xi - x1)
            distances.append(distance)

        # Find the point with maximum distance (elbow)
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]

    def _optimize_spectral_clustering(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        max_clusters: int,
        clustering_options: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Optimize spectral clustering parameters and find optimal k.

        Uses auto parameter information to determine which parameters to optimize.
        """

        # Get auto parameter information from clustering_options
        auto_params = clustering_options.get("_auto_params", {}).get("spectral", {})

        # Default values from sklearn
        sklearn_defaults = {
            "affinity": "rbf",
            "assign_labels": "kmeans",
            "n_neighbors": 10,
            "gamma": None,
        }

        # Determine which parameters to optimize based on auto_params
        optimize_affinity = auto_params.get("affinity", False)
        optimize_assign_labels = auto_params.get("assign_labels", False)
        optimize_n_neighbors = auto_params.get("n_neighbors", False)
        optimize_gamma = auto_params.get("gamma", False)

        # Get fixed values (user-provided parameters or defaults for non-auto)
        fixed_affinity = clustering_options.get("affinity", sklearn_defaults["affinity"])
        fixed_assign_labels = clustering_options.get(
            "assign_labels", sklearn_defaults["assign_labels"]
        )
        fixed_n_neighbors = clustering_options.get("n_neighbors", sklearn_defaults["n_neighbors"])
        fixed_gamma = clustering_options.get("gamma", sklearn_defaults["gamma"])

        # Build candidate parameter combinations (only for parameters not fixed by user)
        n = len(embeddings)

        # Affinity candidates
        if optimize_affinity:
            candidate_affinities = ["rbf", "nearest_neighbors"]
        else:
            candidate_affinities = [fixed_affinity]

        # Gamma candidates for RBF (only if not fixed by user)
        if optimize_gamma and (optimize_affinity or fixed_affinity == "rbf"):
            candidate_gammas = [None]  # sklearn auto
            feature_variance = np.var(embeddings, axis=0).mean()
            if feature_variance > 0:
                scale = 1.0 / (2 * feature_variance)
                candidate_gammas.extend([scale * f for f in [0.5, 1.0, 2.0]])
        else:
            candidate_gammas = [fixed_gamma]

        # n_neighbors candidates (only if not fixed by user)
        if optimize_n_neighbors:
            candidate_n_neighbors = [10, max(5, min(20, int(0.1 * n)))]
            candidate_n_neighbors = sorted(set(candidate_n_neighbors))
        else:
            candidate_n_neighbors = [fixed_n_neighbors]

        # assign_labels candidates (only if not fixed by user)
        if optimize_assign_labels:
            candidate_assign_labels = ["kmeans"]  # Focus on the more reliable option
        else:
            candidate_assign_labels = [fixed_assign_labels]

        # Print optimization strategy
        optimizing = []
        if optimize_affinity:
            optimizing.append("affinity")
        if optimize_gamma:
            optimizing.append("gamma")
        if optimize_n_neighbors:
            optimizing.append("n_neighbors")
        if optimize_assign_labels:
            optimizing.append("assign_labels")

        if optimizing:
            print(f"  Optimizing spectral parameters: {', '.join(optimizing)}")
        else:
            print("  Using user-provided spectral parameters (no optimization)")
            return self._simple_k_optimization_spectral(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # k values to test
        max_possible_clusters = min(max_clusters + 1, len(embeddings))
        if max_possible_clusters <= 2:
            cluster_range = [2]
        else:
            cluster_range = range(2, max_possible_clusters)

        best_score = -np.inf
        best_params = None
        metrics_by_k = {}
        all_results = {}  # Store all parameter combinations

        # Grid search over parameter combinations
        for affinity in candidate_affinities:
            for assign_labels in candidate_assign_labels:

                if affinity == "rbf":
                    gamma_candidates = candidate_gammas
                    n_neighbors_candidates = [10]  # Not used for RBF, placeholder value
                else:  # nearest_neighbors
                    gamma_candidates = [None]  # Not used for nearest_neighbors
                    n_neighbors_candidates = candidate_n_neighbors

                for gamma in gamma_candidates:
                    for n_neighbors in n_neighbors_candidates:

                        # Build parameter dict
                        params = {
                            "affinity": affinity,
                            "assign_labels": assign_labels,
                        }
                        if affinity == "rbf" and gamma is not None:
                            params["gamma"] = gamma
                        elif affinity == "nearest_neighbors":
                            params["n_neighbors"] = n_neighbors

                        # Test this parameter combination across different k values
                        param_best_score = -np.inf
                        param_best_k = 2

                        for k in cluster_range:
                            try:
                                cluster_labels = self.perform_clustering(
                                    embeddings, method="spectral", n_clusters=k, **params
                                )
                                metrics = self.evaluate_clustering(
                                    cluster_labels, true_labels, embeddings
                                )

                                # Add inertia for elbow method
                                inertia = self._calculate_inertia(embeddings, cluster_labels)
                                metrics["inertia"] = inertia

                                # Score using only internal metrics
                                validity = 1 if len(np.unique(cluster_labels)) >= 2 else 0
                                silhouette = max(metrics.get("silhouette_score", -1.0), -1.0)
                                calinski = metrics.get("calinski_harabasz_score", 0.0)
                                davies_bouldin = metrics.get("davies_bouldin_score", float("inf"))

                                score = (
                                    validity * 2.0
                                    + silhouette * 1.0
                                    + min(calinski / 1000.0, 1.0)
                                    - min(davies_bouldin / 10.0, 1.0)
                                )

                                # Track best for this parameter combination
                                if score > param_best_score:
                                    param_best_score = score
                                    param_best_k = k

                                # Store metrics (will be overwritten for each k, but that's fine)
                                metrics_by_k[k] = metrics

                                # Track global best
                                if score > best_score:
                                    best_score = score
                                    best_params = (params.copy(), k)

                            except Exception as e:
                                print(
                                    f"Spectral clustering failed with params {params}, k={k}: {e}"
                                )
                                continue

                        # Store this parameter combination's best result
                        param_key = f"{affinity}_{assign_labels}_{gamma}_{n_neighbors}"
                        all_results[param_key] = (params, param_best_k, param_best_score)

        if best_params is None:
            # Fallback to simple k optimization with default parameters
            print(
                "Warning: Spectral parameter optimization failed, falling back to k optimization only"
            )
            return self._simple_k_optimization_spectral(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # Update clustering_options with best parameters
        best_param_dict, best_k_value = best_params
        clustering_options.update(best_param_dict)

        # Create best_k dict with the optimal k for each metric selection method
        # Since we optimized globally, use the same k for all methods
        best_k = {
            "silhouette": best_k_value,
            "calinski_harabasz": best_k_value,
            "davies_bouldin": best_k_value,
            "elbow": best_k_value,
        }

        # If we have inertia values, also compute elbow method
        if metrics_by_k and any("inertia" in metrics_by_k[k] for k in metrics_by_k):
            k_values = sorted([k for k in metrics_by_k.keys() if "inertia" in metrics_by_k[k]])
            if len(k_values) >= 3:
                inertia_values = [metrics_by_k[k]["inertia"] for k in k_values]
                elbow_k = self._find_elbow_point(k_values, inertia_values)
                best_k["elbow"] = elbow_k

        print(f"Best spectral parameters: {best_param_dict}, k={best_k_value}")
        return best_k, metrics_by_k

    def _simple_k_optimization_spectral(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        max_clusters: int,
        clustering_options: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Fallback: simple k optimization for spectral clustering with default parameters."""

        max_possible_clusters = min(max_clusters + 1, len(embeddings))
        if max_possible_clusters <= 2:
            cluster_range = [2]
        else:
            cluster_range = range(2, max_possible_clusters)

        metrics_by_k = {}

        for k in cluster_range:
            try:
                cluster_labels = self.perform_clustering(
                    embeddings, method="spectral", n_clusters=k, **clustering_options
                )
                metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)

                # Add inertia for elbow method
                inertia = self._calculate_inertia(embeddings, cluster_labels)
                metrics["inertia"] = inertia

                metrics_by_k[k] = metrics
            except Exception as e:
                print(f"Spectral clustering failed with k={k}: {e}")
                continue

        if not metrics_by_k:
            # Last resort - just return something
            best_k = {"silhouette": 2, "calinski_harabasz": 2, "davies_bouldin": 2, "elbow": 2}
            metrics_by_k = {
                2: {
                    "silhouette_score": 0,
                    "calinski_harabasz_score": 0,
                    "davies_bouldin_score": float("inf"),
                }
            }
            return best_k, metrics_by_k

        # Find best k for different metrics
        best_k = {}
        best_k["silhouette"] = max(cluster_range, key=lambda k: metrics_by_k[k]["silhouette_score"])
        best_k["calinski_harabasz"] = max(
            cluster_range, key=lambda k: metrics_by_k[k]["calinski_harabasz_score"]
        )
        best_k["davies_bouldin"] = min(
            cluster_range, key=lambda k: metrics_by_k[k]["davies_bouldin_score"]
        )

        # Add elbow method
        if any("inertia" in metrics_by_k[k] for k in cluster_range):
            inertia_values = [metrics_by_k[k].get("inertia", 0) for k in cluster_range]
            if any(inertia_values):
                best_k["elbow"] = self._find_elbow_point(list(cluster_range), inertia_values)
            else:
                best_k["elbow"] = cluster_range[0]

        return best_k, metrics_by_k

    def _optimize_hierarchical_clustering(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        max_clusters: int,
        clustering_options: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Optimize hierarchical clustering by testing different linkage/metric combinations and k values.

        Uses auto parameter information to determine which parameters to optimize.
        """

        # Get auto parameter information from clustering_options
        auto_params = clustering_options.get("_auto_params", {}).get("hierarchical", {})

        # Default values from sklearn
        sklearn_defaults = {"linkage": "complete", "metric": "euclidean"}

        # Determine which parameters to optimize based on auto_params
        optimize_linkage = auto_params.get("linkage", False)
        optimize_metric = auto_params.get("metric", False)

        # Get fixed values (user-provided parameters or defaults for non-auto)
        fixed_linkage = clustering_options.get("linkage", sklearn_defaults["linkage"])
        fixed_metric = clustering_options.get("metric", sklearn_defaults["metric"])

        # Build candidate parameter combinations (only for parameters in auto mode)
        if optimize_linkage:
            candidate_linkages = ["ward", "complete", "average", "single"]
        else:
            candidate_linkages = [fixed_linkage]

        if optimize_metric:
            # For ward linkage, only euclidean is valid
            candidate_metrics = ["euclidean", "manhattan", "cosine"]
        else:
            candidate_metrics = [fixed_metric]

        # Print optimization strategy
        optimizing = []
        if optimize_linkage:
            optimizing.append("linkage")
        if optimize_metric:
            optimizing.append("metric")

        if optimizing:
            print(f"  Optimizing hierarchical parameters: {', '.join(optimizing)}")
        else:
            print("  Using user-provided hierarchical parameters (no optimization)")
            return self._simple_k_optimization_hierarchical(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # k values to test
        max_possible_clusters = min(max_clusters + 1, len(embeddings))
        if max_possible_clusters <= 2:
            cluster_range = [2]
        else:
            cluster_range = range(2, max_possible_clusters)

        best_score = -np.inf
        best_params = None
        metrics_by_k = {}

        # Grid search over parameter combinations
        for linkage in candidate_linkages:
            for metric in candidate_metrics:

                # Ward linkage only works with euclidean metric
                if linkage == "ward" and metric != "euclidean":
                    continue

                # Build parameter dict
                params = {
                    "linkage": linkage,
                    "metric": metric,
                }

                # Test this parameter combination across different k values
                param_best_score = -np.inf
                param_best_k = 2

                for k in cluster_range:
                    try:
                        cluster_labels = self.perform_clustering(
                            embeddings, method="hierarchical", n_clusters=k, **params
                        )
                        metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)

                        # Add inertia for elbow method
                        inertia = self._calculate_wcss(embeddings, cluster_labels)
                        metrics["inertia"] = inertia

                        # Score using only internal metrics
                        validity = 1 if len(np.unique(cluster_labels)) >= 2 else 0
                        silhouette = max(metrics.get("silhouette_score", -1.0), -1.0)
                        calinski = metrics.get("calinski_harabasz_score", 0.0)
                        davies_bouldin = metrics.get("davies_bouldin_score", float("inf"))

                        score = (
                            validity * 2.0
                            + silhouette * 1.0
                            + min(calinski / 1000.0, 1.0)
                            - min(davies_bouldin / 10.0, 1.0)
                        )

                        # Track best for this parameter combination
                        if score > param_best_score:
                            param_best_score = score
                            param_best_k = k

                        # Store metrics (will be overwritten for each k, but that's fine)
                        metrics_by_k[k] = metrics

                        # Track global best
                        if score > best_score:
                            best_score = score
                            best_params = (params.copy(), k)

                    except Exception as e:
                        print(f"Hierarchical clustering failed with params {params}, k={k}: {e}")
                        continue

        if best_params is None:
            # Fallback to simple k optimization with default parameters
            print(
                "Warning: Hierarchical parameter optimization failed, falling back to k optimization only"
            )
            return self._simple_k_optimization_hierarchical(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # Update clustering_options with best parameters
        best_param_dict, best_k_value = best_params
        clustering_options.update(best_param_dict)

        # Create best_k dict with the optimal k for each metric selection method
        best_k = {
            "silhouette": best_k_value,
            "calinski_harabasz": best_k_value,
            "davies_bouldin": best_k_value,
            "elbow": best_k_value,
        }

        # If we have inertia values, also compute elbow method
        if metrics_by_k and any("inertia" in metrics_by_k[k] for k in metrics_by_k):
            k_values = sorted([k for k in metrics_by_k.keys() if "inertia" in metrics_by_k[k]])
            if len(k_values) >= 3:
                inertia_values = [metrics_by_k[k]["inertia"] for k in k_values]
                elbow_k = self._find_elbow_point(k_values, inertia_values)
                best_k["elbow"] = elbow_k

        print(f"Best hierarchical parameters: {best_param_dict}, k={best_k_value}")
        return best_k, metrics_by_k

    def _simple_k_optimization_hierarchical(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        max_clusters: int,
        clustering_options: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Fallback: simple k optimization for hierarchical clustering with default parameters."""

        max_possible_clusters = min(max_clusters + 1, len(embeddings))
        if max_possible_clusters <= 2:
            cluster_range = [2]
        else:
            cluster_range = range(2, max_possible_clusters)

        metrics_by_k = {}

        for k in cluster_range:
            try:
                cluster_labels = self.perform_clustering(
                    embeddings, method="hierarchical", n_clusters=k, **clustering_options
                )
                metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)

                # Add inertia for elbow method
                inertia = self._calculate_wcss(embeddings, cluster_labels)
                metrics["inertia"] = inertia

                metrics_by_k[k] = metrics
            except Exception as e:
                print(f"Hierarchical clustering failed with k={k}: {e}")
                continue

        if not metrics_by_k:
            # Last resort - just return something
            best_k = {"silhouette": 2, "calinski_harabasz": 2, "davies_bouldin": 2, "elbow": 2}
            metrics_by_k = {
                2: {
                    "silhouette_score": 0,
                    "calinski_harabasz_score": 0,
                    "davies_bouldin_score": float("inf"),
                }
            }
            return best_k, metrics_by_k

        # Find best k for different metrics
        best_k = {}
        best_k["silhouette"] = max(cluster_range, key=lambda k: metrics_by_k[k]["silhouette_score"])
        best_k["calinski_harabasz"] = max(
            cluster_range, key=lambda k: metrics_by_k[k]["calinski_harabasz_score"]
        )
        best_k["davies_bouldin"] = min(
            cluster_range, key=lambda k: metrics_by_k[k]["davies_bouldin_score"]
        )

        # Add elbow method
        if any("inertia" in metrics_by_k[k] for k in cluster_range):
            inertia_values = [metrics_by_k[k].get("inertia", 0) for k in cluster_range]
            if any(inertia_values):
                best_k["elbow"] = self._find_elbow_point(list(cluster_range), inertia_values)
            else:
                best_k["elbow"] = cluster_range[0]

        return best_k, metrics_by_k

    def _optimize_kmeans_clustering(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        max_clusters: int,
        clustering_options: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Optimize k-means clustering by testing different initialization strategies and k values.

        Uses auto parameter information to determine which parameters to optimize.
        """

        # Get auto parameter information from clustering_options
        auto_params = clustering_options.get("_auto_params", {}).get("kmeans", {})

        # Default values from sklearn
        sklearn_defaults = {"init": "k-means++", "max_iter": 300, "random_state": 42}

        # Determine which parameters to optimize based on auto_params
        optimize_init = auto_params.get("init", False)
        optimize_max_iter = auto_params.get("max_iter", False)

        # Get fixed values (user-provided parameters or defaults for non-auto)
        fixed_init = clustering_options.get("init", sklearn_defaults["init"])
        fixed_max_iter = clustering_options.get("max_iter", sklearn_defaults["max_iter"])

        # Build candidate parameter combinations (only for parameters in auto mode)
        if optimize_init:
            candidate_inits = ["k-means++", "random"]
        else:
            candidate_inits = [fixed_init]

        if optimize_max_iter:
            candidate_max_iters = [100, 200, 300, 500]
        else:
            candidate_max_iters = [fixed_max_iter]

        # Print optimization strategy
        optimizing = []
        if optimize_init:
            optimizing.append("init")
        if optimize_max_iter:
            optimizing.append("max_iter")

        if optimizing:
            print(f"  Optimizing k-means parameters: {', '.join(optimizing)}")
        else:
            print("  Using user-provided k-means parameters (no optimization)")
            return self._simple_k_optimization_kmeans(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # Fixed random state for reproducibility
        fixed_random_state = clustering_options.get(
            "random_state", sklearn_defaults["random_state"]
        )

        # k values to test
        max_possible_clusters = min(max_clusters + 1, len(embeddings))
        if max_possible_clusters <= 2:
            cluster_range = [2]
        else:
            cluster_range = range(2, max_possible_clusters)

        best_score = -np.inf
        best_params = None
        metrics_by_k = {}

        # Grid search over parameter combinations
        for init in candidate_inits:
            for max_iter in candidate_max_iters:

                # Build parameter dict
                params = {
                    "init": init,
                    "max_iter": max_iter,
                    "random_state": fixed_random_state,  # Keep reproducible
                }

                # Test this parameter combination across different k values
                param_best_score = -np.inf
                param_best_k = 2

                for k in cluster_range:
                    try:
                        cluster_labels = self.perform_clustering(
                            embeddings, method="kmeans", n_clusters=k, **params
                        )
                        metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)

                        # Add inertia for elbow method
                        inertia = self._calculate_inertia(embeddings, cluster_labels)
                        metrics["inertia"] = inertia

                        # Score using only internal metrics
                        validity = 1 if len(np.unique(cluster_labels)) >= 2 else 0
                        silhouette = max(metrics.get("silhouette_score", -1.0), -1.0)
                        calinski = metrics.get("calinski_harabasz_score", 0.0)
                        davies_bouldin = metrics.get("davies_bouldin_score", float("inf"))

                        score = (
                            validity * 2.0
                            + silhouette * 1.0
                            + min(calinski / 1000.0, 1.0)
                            - min(davies_bouldin / 10.0, 1.0)
                        )

                        # Track best for this parameter combination
                        if score > param_best_score:
                            param_best_score = score
                            param_best_k = k

                        # Store metrics (will be overwritten for each k, but that's fine)
                        metrics_by_k[k] = metrics

                        # Track global best
                        if score > best_score:
                            best_score = score
                            best_params = (params.copy(), k)

                    except Exception as e:
                        print(f"K-means clustering failed with params {params}, k={k}: {e}")
                        continue

        if best_params is None:
            # Fallback to simple k optimization with default parameters
            print(
                "Warning: K-means parameter optimization failed, falling back to k optimization only"
            )
            return self._simple_k_optimization_kmeans(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # Update clustering_options with best parameters
        best_param_dict, best_k_value = best_params
        clustering_options.update(best_param_dict)

        # Create best_k dict with the optimal k for each metric selection method
        best_k = {
            "silhouette": best_k_value,
            "calinski_harabasz": best_k_value,
            "davies_bouldin": best_k_value,
            "elbow": best_k_value,
        }

        # If we have inertia values, also compute elbow method
        if metrics_by_k and any("inertia" in metrics_by_k[k] for k in metrics_by_k):
            k_values = sorted([k for k in metrics_by_k.keys() if "inertia" in metrics_by_k[k]])
            if len(k_values) >= 3:
                inertia_values = [metrics_by_k[k]["inertia"] for k in k_values]
                elbow_k = self._find_elbow_point(k_values, inertia_values)
                best_k["elbow"] = elbow_k

        print(f"Best k-means parameters: {best_param_dict}, k={best_k_value}")
        return best_k, metrics_by_k

    def _simple_k_optimization_kmeans(
        self,
        embeddings: np.ndarray,
        true_labels: np.ndarray,
        max_clusters: int,
        clustering_options: Dict[str, Any],
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Fallback: simple k optimization for k-means clustering with default parameters."""

        max_possible_clusters = min(max_clusters + 1, len(embeddings))
        if max_possible_clusters <= 2:
            cluster_range = [2]
        else:
            cluster_range = range(2, max_possible_clusters)

        metrics_by_k = {}

        for k in cluster_range:
            try:
                cluster_labels = self.perform_clustering(
                    embeddings, method="kmeans", n_clusters=k, **clustering_options
                )
                metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)

                # Add inertia for elbow method
                inertia = self._calculate_inertia(embeddings, cluster_labels)
                metrics["inertia"] = inertia

                metrics_by_k[k] = metrics
            except Exception as e:
                print(f"K-means clustering failed with k={k}: {e}")
                continue

        if not metrics_by_k:
            # Last resort - just return something
            best_k = {"silhouette": 2, "calinski_harabasz": 2, "davies_bouldin": 2, "elbow": 2}
            metrics_by_k = {
                2: {
                    "silhouette_score": 0,
                    "calinski_harabasz_score": 0,
                    "davies_bouldin_score": float("inf"),
                }
            }
            return best_k, metrics_by_k

        # Find best k for different metrics
        best_k = {}
        best_k["silhouette"] = max(cluster_range, key=lambda k: metrics_by_k[k]["silhouette_score"])
        best_k["calinski_harabasz"] = max(
            cluster_range, key=lambda k: metrics_by_k[k]["calinski_harabasz_score"]
        )
        best_k["davies_bouldin"] = min(
            cluster_range, key=lambda k: metrics_by_k[k]["davies_bouldin_score"]
        )

        # Add elbow method
        if any("inertia" in metrics_by_k[k] for k in cluster_range):
            inertia_values = [metrics_by_k[k].get("inertia", 0) for k in cluster_range]
            if any(inertia_values):
                best_k["elbow"] = self._find_elbow_point(list(cluster_range), inertia_values)
            else:
                best_k["elbow"] = cluster_range[0]

        return best_k, metrics_by_k

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
        clustering_options: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, int], Dict[int, Dict[str, float]]]:
        """Find optimal number of clusters using multiple criteria. For dbscan/hdbscan, just evaluate with provided parameters."""

        print(f"Finding optimal number of clusters for {method}...")

        if clustering_options is None:
            clustering_options = {}

        # For density-based methods, handle specially
        if method == "hdbscan":
            # Get auto parameter information from clustering_options
            auto_params = clustering_options.get("_auto_params", {}).get("hdbscan", {})

            # Check which parameters need optimization based on auto_params
            min_cluster_size_default = 5
            min_samples_default = None

            user_min_cluster_size = clustering_options.get(
                "min_cluster_size", min_cluster_size_default
            )
            user_min_samples = clustering_options.get("min_samples", min_samples_default)

            # Determine which parameters to optimize based on auto_params
            optimize_min_cluster_size = auto_params.get("min_cluster_size", False)
            optimize_min_samples = auto_params.get("min_samples", False)

            params_to_optimize = []
            if optimize_min_cluster_size:
                params_to_optimize.append("min_cluster_size")
            if optimize_min_samples:
                params_to_optimize.append("min_samples")

            if not params_to_optimize:
                print(f"  Using user-provided HDBSCAN parameters (no optimization)")
                # Just run with user parameters and return
                options = clustering_options.copy()
                cluster_labels = self.perform_clustering(embeddings, method=method, **options)
                metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)
                unique_labels = set(cluster_labels)
                n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                metrics_by_k = {n_clusters_found: metrics}
                best_k = {
                    m: n_clusters_found
                    for m in ["silhouette", "calinski_harabasz", "davies_bouldin", "elbow"]
                }
                return best_k, metrics_by_k
            else:
                print(f"  Optimizing HDBSCAN parameters: {', '.join(params_to_optimize)}")

            # Optimize HDBSCAN parameters: min_cluster_size and min_samples
            n = len(embeddings)

            # Build candidate ranges based on what needs optimization
            if optimize_min_cluster_size:
                candidate_min_cluster_sizes = sorted(
                    {
                        2,
                        3,
                        5,
                        min_cluster_size_default,
                        max(2, int(0.01 * n)),
                        max(2, int(0.02 * n)),
                        max(2, int(0.05 * n)),
                    }
                )
                candidate_min_cluster_sizes = [
                    s for s in candidate_min_cluster_sizes if s <= max(2, int(0.1 * n))
                ]
            else:
                candidate_min_cluster_sizes = [user_min_cluster_size]  # Use user-provided value

            if optimize_min_samples:
                candidate_min_samples = [None]  # None means use min_cluster_size
                if min_samples_default is not None:
                    candidate_min_samples.extend([1, min_samples_default, max(1, int(0.01 * n))])
            else:
                candidate_min_samples = [user_min_samples]  # Use user-provided value

            best_score = -np.inf
            best_result = None
            metrics_by_k = {}

            for min_cluster_size in candidate_min_cluster_sizes:
                for min_samples in candidate_min_samples:
                    try:
                        options = clustering_options.copy()
                        options["min_cluster_size"] = min_cluster_size
                        if min_samples is not None:
                            options["min_samples"] = min_samples

                        cluster_labels = self.perform_clustering(
                            embeddings, method=method, **options
                        )
                        metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)
                        unique_labels = set(cluster_labels)
                        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)

                        # Score using only internal metrics
                        validity = 1 if n_clusters_found >= 2 else 0
                        silhouette = max(metrics.get("silhouette_score", -1.0), -1.0)
                        calinski = metrics.get("calinski_harabasz_score", 0.0)
                        davies_bouldin = metrics.get("davies_bouldin_score", float("inf"))

                        score = (
                            validity * 2.0
                            + silhouette * 1.0
                            + min(calinski / 1000.0, 1.0)
                            - min(davies_bouldin / 10.0, 1.0)
                        )

                        if score > best_score:
                            best_score = score
                            best_result = (
                                cluster_labels,
                                metrics,
                                n_clusters_found,
                                min_cluster_size,
                                min_samples,
                            )

                        metrics_by_k[n_clusters_found] = metrics

                    except Exception as e:
                        print(
                            f"HDBSCAN failed with min_cluster_size={min_cluster_size}, min_samples={min_samples}: {e}"
                        )
                        continue

            if best_result is None:
                # Fallback to default parameters
                cluster_labels = self.perform_clustering(
                    embeddings, method=method, **clustering_options
                )
                metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)
                unique_labels = set(cluster_labels)
                n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
                best_result = (
                    cluster_labels,
                    metrics,
                    n_clusters_found,
                    min_cluster_size_default,
                    min_samples_default,
                )
                metrics_by_k[n_clusters_found] = metrics

            # Update clustering_options with best parameters
            clustering_options["min_cluster_size"] = best_result[3]
            if best_result[4] is not None:
                clustering_options["min_samples"] = best_result[4]

            k_key = max(best_result[2], 0)
            best_k = {
                m: k_key
                for m in [
                    "silhouette",
                    "calinski_harabasz",
                    "davies_bouldin",
                    "elbow",  # Include elbow even though not applicable to HDBSCAN
                ]
            }
            return best_k, metrics_by_k

        if method == "dbscan":
            # Get auto parameter information from clustering_options
            auto_params = clustering_options.get("_auto_params", {}).get("dbscan", {})

            # Check which parameters need optimization based on auto_params
            eps_default = 0.5
            min_samples_default = 5

            user_eps = clustering_options.get("eps", eps_default)
            user_min_samples = clustering_options.get("min_samples", min_samples_default)

            # Determine which parameters to optimize based on auto_params
            optimize_eps = auto_params.get("eps", False)
            optimize_min_samples = auto_params.get("min_samples", False)

            params_to_optimize = []
            if optimize_eps:
                params_to_optimize.append("eps")
            if optimize_min_samples:
                params_to_optimize.append("min_samples")

            if not params_to_optimize:
                print(f"  Using user-provided DBSCAN parameters (no optimization)")
                # Just run with user parameters and return
                labels = self.perform_clustering(
                    embeddings, method="dbscan", eps=user_eps, min_samples=user_min_samples
                )
                uniq = set(labels)
                n_c = len(uniq) - (1 if -1 in uniq else 0)
                metrics = self.evaluate_clustering(labels, true_labels, embeddings)
                metrics_by_k = {n_c: metrics}
                best_k = {
                    m: n_c for m in ["silhouette", "calinski_harabasz", "davies_bouldin", "elbow"]
                }
                return best_k, metrics_by_k
            else:
                print(f"  Optimizing DBSCAN parameters: {', '.join(params_to_optimize)}")

            # Search eps and min_samples using k-distance quantiles
            # Build candidate grids
            n = len(embeddings)

            # Generate candidates based on what needs optimization
            if optimize_min_samples:
                candidate_min_samples = sorted(
                    {2, 3, 4, 5, min_samples_default, max(2, int(0.01 * n)), max(2, int(0.02 * n))}
                )
                candidate_min_samples = [
                    m for m in candidate_min_samples if m <= max(2, int(0.05 * n))
                ]
            else:
                candidate_min_samples = [user_min_samples]  # Use user-provided value

            if optimize_eps:
                # Compute k-distance for a base k (use max of candidates)
                k_for_knn = min(max(candidate_min_samples), max(2, min(50, n - 1)))
                try:
                    nbrs = NearestNeighbors(n_neighbors=k_for_knn).fit(embeddings)
                    distances, _ = nbrs.kneighbors(embeddings)
                    # Use the distance to the k-th neighbor
                    kth_dist = distances[:, -1]
                    quantiles = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
                    candidate_eps = sorted(
                        {float(np.quantile(kth_dist, q)) for q in quantiles} | {eps_default}
                    )
                except Exception:
                    # Fallback to a simple eps list around default
                    candidate_eps = sorted({eps_default * f for f in [0.5, 0.75, 1.0, 1.25, 1.5]})
            else:
                candidate_eps = [user_eps]  # Use user-provided value

            metrics_by_k: Dict[int, Dict[str, float]] = {}
            # Track best by a composite score preferring valid clusterings
            best_score = -np.inf
            best_result = None  # (labels, metrics, n_clusters, eps, min_samples)

            for ms in candidate_min_samples:
                for eps in candidate_eps:
                    labels = self.perform_clustering(
                        embeddings, method="dbscan", eps=eps, min_samples=int(ms)
                    )
                    # Count clusters excluding noise
                    uniq = set(labels)
                    n_c = len(uniq) - (1 if -1 in uniq else 0)
                    metrics = self.evaluate_clustering(labels, true_labels, embeddings)
                    # Composite score using only internal metrics (no ground truth)
                    validity = 1 if n_c >= 2 else 0  # Prefer valid clusterings
                    silhouette = max(metrics.get("silhouette_score", -1.0), -1.0)
                    calinski = metrics.get("calinski_harabasz_score", 0.0)
                    davies_bouldin = metrics.get("davies_bouldin_score", float("inf"))

                    # Composite score: validity + silhouette + normalized calinski - davies_bouldin
                    score = (
                        validity * 2.0  # Strong preference for valid clusterings
                        + silhouette * 1.0  # Silhouette score weight
                        + min(calinski / 1000.0, 1.0)  # Normalized Calinski-Harabasz (cap at 1.0)
                        - min(davies_bouldin / 10.0, 1.0)  # Normalized Davies-Bouldin penalty
                    )
                    # Prefer more clusters if tie on score
                    tie_break = n_c
                    if score > best_score or (
                        np.isclose(score, best_score)
                        and tie_break > (best_result[2] if best_result else -1)
                    ):
                        best_score = score
                        best_result = (labels, metrics, n_c, eps, int(ms))

                    # Store metrics keyed by number of clusters seen for overview (last wins per k)
                    metrics_by_k[n_c] = metrics

            if best_result is None:
                # As a last resort, run default
                labels = self.perform_clustering(
                    embeddings, method="dbscan", eps=eps_default, min_samples=min_samples_default
                )
                uniq = set(labels)
                n_c = len(uniq) - (1 if -1 in uniq else 0)
                metrics = self.evaluate_clustering(labels, true_labels, embeddings)
                best_result = (labels, metrics, n_c, eps_default, min_samples_default)
                metrics_by_k[n_c] = metrics

            # Update options so downstream run uses the selected params
            clustering_options["eps"], clustering_options["min_samples"] = (
                best_result[3],
                best_result[4],
            )
            n_clusters_found = max(best_result[2], 0)
            best_k = {
                m: n_clusters_found
                for m in [
                    "silhouette",
                    "calinski_harabasz",
                    "davies_bouldin",
                    "elbow",  # Include elbow even though not applicable to DBSCAN
                ]
            }
            return best_k, metrics_by_k

        # Special optimization for spectral clustering
        if method == "spectral":
            return self._optimize_spectral_clustering(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # Special optimization for hierarchical clustering
        if method == "hierarchical":
            return self._optimize_hierarchical_clustering(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # Special optimization for k-means clustering
        if method == "kmeans":
            return self._optimize_kmeans_clustering(
                embeddings, true_labels, max_clusters, clustering_options
            )

        # Ensure we have a valid range for clustering
        max_possible_clusters = min(max_clusters + 1, len(embeddings))
        if max_possible_clusters <= 2:
            print(
                f"Warning: Not enough data points ({len(embeddings)}) for optimization. Using 2 clusters."
            )
            best_k = {
                "silhouette": 2,
                "calinski_harabasz": 2,
                "davies_bouldin": 2,
                "elbow": 2,
            }
            metrics_by_k = {
                2: self.evaluate_clustering(
                    self.perform_clustering(
                        embeddings, method=method, n_clusters=2, **clustering_options
                    ),
                    true_labels,
                    embeddings,
                )
            }
            return best_k, metrics_by_k

        cluster_range = range(2, max_possible_clusters)
        metrics_by_k = {}

        for k in cluster_range:
            print(f"Testing k={k}")
            cluster_labels = self.perform_clustering(
                embeddings, method=method, n_clusters=k, **clustering_options
            )
            metrics = self.evaluate_clustering(cluster_labels, true_labels, embeddings)

            # Add inertia/WCSS for elbow method (only for kmeans-like methods)
            if method in ["kmeans", "spectral"]:
                # For kmeans, we can calculate inertia (WCSS)
                inertia = self._calculate_inertia(embeddings, cluster_labels)
                metrics["inertia"] = inertia
            elif method == "hierarchical":
                # For hierarchical, use within-cluster sum of squares approximation
                wcss = self._calculate_wcss(embeddings, cluster_labels)
                metrics["inertia"] = wcss

            metrics_by_k[k] = metrics

        # Find best k for different metrics (only internal metrics - no ground truth)
        best_k = {}
        best_k["silhouette"] = max(cluster_range, key=lambda k: metrics_by_k[k]["silhouette_score"])
        best_k["calinski_harabasz"] = max(
            cluster_range, key=lambda k: metrics_by_k[k]["calinski_harabasz_score"]
        )
        best_k["davies_bouldin"] = min(
            cluster_range, key=lambda k: metrics_by_k[k]["davies_bouldin_score"]
        )

        # Add elbow method for methods that support it
        if method in ["kmeans", "hierarchical", "spectral"]:
            inertia_values = [metrics_by_k[k].get("inertia", 0) for k in cluster_range]
            if any(inertia_values):  # Only if we have inertia values
                best_k["elbow"] = self._find_elbow_point(list(cluster_range), inertia_values)
            else:
                best_k["elbow"] = cluster_range[0]  # Fallback

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
        """Plot confusion matrix (truth table) with true labels as rows and cluster labels as columns."""
        from scipy.optimize import linear_sum_assignment

        # Get unique labels and clusters that actually exist in the data
        unique_true_labels = np.unique(true_labels)
        unique_cluster_labels = np.unique(cluster_labels)

        # Create confusion matrix with explicit labels for both rows and columns
        # This ensures we get the right dimensions: true_labels x cluster_labels
        cm_correct = np.zeros((len(unique_true_labels), len(unique_cluster_labels)), dtype=int)

        # Fill the correct confusion matrix
        for i, true_label in enumerate(unique_true_labels):
            for j, cluster_label in enumerate(unique_cluster_labels):
                # Count how many samples have this true_label and cluster_label combination
                count = np.sum((true_labels == true_label) & (cluster_labels == cluster_label))
                cm_correct[i, j] = count

        # Sort rows and columns to maximize diagonal values using Hungarian algorithm
        # For rectangular matrices, we need to handle this carefully
        min_dim = min(cm_correct.shape[0], cm_correct.shape[1])

        if min_dim > 1:
            # Create a square matrix for the Hungarian algorithm by taking the top-left submatrix
            # or padding if needed
            if cm_correct.shape[0] == cm_correct.shape[1]:
                # Square matrix - use directly
                cost_matrix = -cm_correct  # Negative because we want to maximize
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
            elif cm_correct.shape[0] < cm_correct.shape[1]:
                # More clusters than true labels - optimize for all true labels
                cost_matrix = -cm_correct
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                # Keep remaining cluster columns in original order
                remaining_cols = [i for i in range(cm_correct.shape[1]) if i not in col_indices]
                col_indices = np.concatenate([col_indices, remaining_cols])
                row_indices = np.concatenate(
                    [row_indices, np.arange(len(row_indices), cm_correct.shape[0])]
                )
            else:
                # More true labels than clusters - optimize for all clusters
                cost_matrix = -cm_correct.T  # Transpose for assignment
                col_indices, row_indices = linear_sum_assignment(cost_matrix)
                # Keep remaining row labels in original order
                remaining_rows = [i for i in range(cm_correct.shape[0]) if i not in row_indices]
                row_indices = np.concatenate([row_indices, remaining_rows])
                col_indices = np.concatenate(
                    [col_indices, np.arange(len(col_indices), cm_correct.shape[1])]
                )

            # Reorder the matrix
            cm_sorted = cm_correct[np.ix_(row_indices, col_indices)]

            # Reorder labels accordingly
            sorted_true_labels = [unique_true_labels[i] for i in row_indices]
            sorted_cluster_labels = [unique_cluster_labels[i] for i in col_indices]
        else:
            # No sorting possible with single row/column
            cm_sorted = cm_correct
            sorted_true_labels = unique_true_labels
            sorted_cluster_labels = unique_cluster_labels

        # Create row labels (true labels) - map indices to actual label names
        row_labels = []
        for label_idx in sorted_true_labels:
            if label_idx < len(label_names):
                row_labels.append(label_names[label_idx])
            else:
                row_labels.append(f"Label {label_idx}")

        # Create column labels (cluster labels)
        col_labels = [f"Cluster {c}" for c in sorted_cluster_labels]

        # Determine figure size based on matrix dimensions
        fig_width = max(8, 2 + len(col_labels) * 0.8)
        fig_height = max(6, 2 + len(row_labels) * 0.5)

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            cm_sorted,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=col_labels,
            yticklabels=row_labels,
        )
        plt.xlabel("Cluster Label (sorted)")
        plt.ylabel("True Label (sorted)")
        plt.title(title + " (sorted)")
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
        """Plot internal metrics vs number of clusters (excludes external/ground truth metrics)."""

        k_values = sorted(metrics_by_k.keys())

        # Only plot internal metrics (avoid using ground truth) + elbow method
        metrics_to_plot = [
            ("silhouette_score", "Silhouette Score"),
            ("calinski_harabasz_score", "Calinski-Harabasz Score"),
            ("davies_bouldin_score", "Davies-Bouldin Score"),
        ]

        # Add inertia plot if available (for elbow method)
        if any("inertia" in metrics_by_k[k] for k in k_values):
            metrics_to_plot.append(("inertia", "Inertia (WCSS) - Elbow Method"))

        # Adjust subplot layout based on number of metrics
        n_metrics = len(metrics_to_plot)
        if n_metrics <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))  # A4 landscape size
            axes = axes.flatten()

        for i, (metric_key, metric_name) in enumerate(metrics_to_plot):
            # Skip if metric not available in all k values
            if not all(metric_key in metrics_by_k[k] for k in k_values):
                continue

            values = [metrics_by_k[k][metric_key] for k in k_values]
            axes[i].plot(k_values, values, "bo-", linewidth=2, markersize=6)
            axes[i].set_xlabel("Number of Clusters")
            axes[i].set_ylabel(metric_name)
            axes[i].set_title(f"{metric_name} vs Number of Clusters")
            axes[i].grid(True, alpha=0.3)

            # Mark the best value
            if metric_key == "davies_bouldin_score":  # Lower is better
                best_k = k_values[np.argmin(values)]
            elif metric_key == "inertia":  # Elbow method - mark elbow point
                # Use the static method directly
                best_k = ClusteringEngine._find_elbow_point(k_values, values)
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

        # Hide unused subplots
        for i in range(len(metrics_to_plot), len(axes)):
            axes[i].set_visible(False)

        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()

    def plot_significance_heatmap(
        self, significance_file: str, output_dir: str, title: str = "Embedding Significance"
    ):
        """Plot significance heatmaps from statistical test results."""
        try:
            df = pd.read_csv(significance_file, sep="\t")
        except Exception as e:
            print(f"Warning: Could not read significance file {significance_file}: {e}")
            return

        # Find corrected p-value column
        pval_col = None
        for col in df.columns:
            if "holm" in col.lower() or "corrected" in col.lower():
                pval_col = col
                break

        if pval_col is None:
            print(f"Warning: No corrected p-value column found in {significance_file}")
            return

        # Calculate mean differences for heatmap values
        if "embedding1_mean" in df.columns and "embedding2_mean" in df.columns:
            df["diff"] = df["embedding1_mean"] - df["embedding2_mean"]
        else:
            print(f"Warning: Mean columns not found in {significance_file}")
            return

        # Create heatmaps for each metric and method
        for metric in df["metric"].unique():
            for method in df["method"].unique():
                sub = df[(df["metric"] == metric) & (df["method"] == method)]
                if sub.empty:
                    continue

                try:
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
                except Exception as e:
                    print(f"Warning: Could not create heatmap for {metric}/{method}: {e}")
                    continue


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
            # Set random seed for reproducibility
            random.seed(run_idx)
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

            # Apply new pipeline if configured, else legacy normalization
            if (
                self.config.norm_center
                or self.config.norm_scale
                or (
                    isinstance(self.config.norm_pca_components, (int, float))
                    and self.config.norm_pca_components != 0
                )
                or (self.config.norm_l2 is False)
                or self.config.normalization_method == "pipeline"
            ):
                emb_matrix, pca_info = EmbeddingNormalizer.normalize_pipeline(
                    emb_matrix,
                    center=self.config.norm_center,
                    scale=self.config.norm_scale,
                    pca_components=self.config.norm_pca_components,
                    l2=self.config.norm_l2,
                )
                # Store PCA info for comments file
                if not hasattr(self, "_pca_info"):
                    self._pca_info = {}
                emb_basename = os.path.splitext(os.path.basename(emb_name))[0]
                self._pca_info[emb_basename] = pca_info
            elif self.config.normalization_method != "none":
                emb_matrix = EmbeddingNormalizer.normalize_embeddings(
                    emb_matrix, self.config.normalization_method
                )

            for method in self.config.methods:
                # Get method-specific options
                method_options = self.config.clustering_options.get(method, {})

                if self.config.n_clusters and method in ["kmeans", "hierarchical"]:
                    cluster_labels = self.clustering_engine.perform_clustering(
                        emb_matrix,
                        method=method,
                        n_clusters=self.config.n_clusters,
                        **method_options,
                    )
                else:
                    best_k, _ = self.clustering_engine.find_optimal_clusters(
                        emb_matrix,
                        sampled_labels,
                        method=method,
                        max_clusters=self.config.max_clusters,
                        clustering_options=method_options,
                    )
                    if method in ["dbscan", "hdbscan"]:
                        cluster_labels = self.clustering_engine.perform_clustering(
                            emb_matrix, method=method, **method_options
                        )
                    else:
                        # Use configurable internal metric for k selection to avoid using ground truth
                        optimal_k = best_k[self.config.k_selection_metric]
                        cluster_labels = self.clustering_engine.perform_clustering(
                            emb_matrix, method=method, n_clusters=optimal_k, **method_options
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

        subsample_results = Parallel(n_jobs=-1, prefer="threads")(
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

        # Validate that the label column exists in metadata
        if self.config.label_column not in metadata.columns:
            available_columns = ", ".join(metadata.columns.tolist())
            raise ValueError(
                f"Label column '{self.config.label_column}' not found in metadata. "
                f"Available columns: {available_columns}"
            )

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

            # Apply normalization: pipeline if configured or requested, else legacy method
            if (
                self.config.norm_center
                or self.config.norm_scale
                or (
                    isinstance(self.config.norm_pca_components, (int, float))
                    and self.config.norm_pca_components != 0
                )
                or (self.config.norm_l2 is False)
                or self.config.normalization_method == "pipeline"
            ):
                print(
                    "Applying pipeline normalization: "
                    f"center={self.config.norm_center}, scale={self.config.norm_scale}, "
                    f"pca_components={self.config.norm_pca_components}, l2={self.config.norm_l2}"
                )
                embedding_matrix, pca_info = EmbeddingNormalizer.normalize_pipeline(
                    embedding_matrix,
                    center=self.config.norm_center,
                    scale=self.config.norm_scale,
                    pca_components=self.config.norm_pca_components,
                    l2=self.config.norm_l2,
                )
                # Store PCA info for comments file
                if not hasattr(self, "_pca_info"):
                    self._pca_info = {}
                self._pca_info["single_embedding"] = pca_info
            elif self.config.normalization_method != "none":
                print(f"Applying {self.config.normalization_method} normalization...")
                embedding_matrix = EmbeddingNormalizer.normalize_embeddings(
                    embedding_matrix, self.config.normalization_method
                )

            # Run clustering for each method
            results = {}
            for method in self.config.methods:
                print(f"\nClustering with {method}...")

                # Get method-specific options and include auto_params information
                method_options = self.config.clustering_options.get(method, {}).copy()
                # Add auto_params information for this method
                if "_auto_params" in self.config.clustering_options:
                    method_options["_auto_params"] = {
                        method: self.config.clustering_options["_auto_params"].get(method, {})
                    }

                if self.config.n_clusters and method in ["kmeans", "hierarchical", "spectral"]:
                    cluster_labels = self.clustering_engine.perform_clustering(
                        embedding_matrix,
                        method=method,
                        n_clusters=self.config.n_clusters,
                        **method_options,
                    )
                    n_clusters = self.config.n_clusters
                else:
                    best_k, optimization_results = self.clustering_engine.find_optimal_clusters(
                        embedding_matrix,
                        true_labels,
                        method=method,
                        max_clusters=self.config.max_clusters,
                        clustering_options=method_options,
                    )

                    if method in ["dbscan", "hdbscan"]:
                        # Use discovered parameters already stored in method_options
                        cluster_labels = self.clustering_engine.perform_clustering(
                            embedding_matrix, method=method, **method_options
                        )
                        # Determine actual number of clusters (exclude noise)
                        uniq = set(cluster_labels)
                        n_clusters = len(uniq) - (1 if -1 in uniq else 0)
                        # Log chosen params for transparency
                        if method == "dbscan":
                            print(
                                f"Selected DBSCAN params: eps={method_options.get('eps')}, min_samples={method_options.get('min_samples')} -> clusters={n_clusters}"
                            )
                        else:
                            print(
                                f"HDBSCAN params: min_cluster_size={method_options.get('min_cluster_size')}, min_samples={method_options.get('min_samples')} -> clusters={n_clusters}"
                            )
                        # Optional: plot overview of metrics vs number of clusters found
                        emb_name = os.path.splitext(os.path.basename(embedding_file))[0]
                        optimization_path = os.path.join(
                            self.config.output_dir, f"cluster_optimization_{emb_name}_{method}.pdf"
                        )
                        if optimization_results:
                            self.visualizer.plot_cluster_optimization(
                                optimization_results, optimization_path
                            )
                    else:
                        # Use configurable internal metric for k selection to avoid using ground truth
                        optimal_k = best_k[self.config.k_selection_metric]
                        print(
                            f"Optimal number of clusters (by {self.config.k_selection_metric}): {optimal_k}"
                        )

                        cluster_labels = self.clustering_engine.perform_clustering(
                            embedding_matrix, method=method, n_clusters=optimal_k, **method_options
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

                # Capture parameters used
                used_params: Dict[str, Any] = {}
                if method == "kmeans":
                    used_params = {
                        "n_clusters": n_clusters,
                        "init": method_options.get("init"),
                        "max_iter": method_options.get("max_iter"),
                    }
                elif method == "hierarchical":
                    used_params = {
                        "n_clusters": n_clusters,
                        "linkage": method_options.get("linkage"),
                        "metric": method_options.get("metric"),
                    }
                elif method == "dbscan":
                    used_params = {
                        "eps": method_options.get("eps"),
                        "min_samples": method_options.get("min_samples"),
                    }
                elif method == "hdbscan":
                    used_params = {
                        "min_cluster_size": method_options.get("min_cluster_size"),
                        "min_samples": method_options.get("min_samples"),
                        "cluster_selection_epsilon": method_options.get(
                            "cluster_selection_epsilon"
                        ),
                    }
                elif method == "spectral":
                    used_params = {
                        "n_clusters": n_clusters,
                        "affinity": method_options.get("affinity"),
                        "assign_labels": method_options.get("assign_labels"),
                        "n_neighbors": method_options.get("n_neighbors"),
                        "gamma": method_options.get("gamma"),
                    }

                results[method] = ClusteringResult(
                    cluster_labels=cluster_labels,
                    n_clusters=n_clusters,
                    metrics=metrics,
                    params=used_params,
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
                # Flatten params into JSON string for readability
                row["params_json"] = json.dumps(convert_numpy_types(result.params))
                results_df.append(row)

            results_df = pd.DataFrame(results_df)
            results_df.to_csv(
                os.path.join(self.config.output_dir, f"{emb_name}_clustering_results.tsv"),
                sep="\t",
                index=False,
            )

        # Create summary file and parameters audit
        summary_results: List[Dict[str, Any]] = []
        params_records: List[Dict[str, Any]] = []
        for embedding_file, emb_result in embedding_results.items():
            emb_name = os.path.splitext(os.path.basename(embedding_file))[0]
            for method, result in emb_result["results"].items():
                # Summary row of metrics
                row = {
                    "embedding": emb_name,
                    "method": method,
                    "n_clusters": result.n_clusters,
                }
                row.update(result.metrics)
                summary_results.append(row)

                # Parameters record
                p_row = {"embedding": emb_name, "method": method}
                p_row.update(result.params)
                params_records.append(p_row)

        if summary_results:
            summary_df = pd.DataFrame(summary_results)
            summary_path = os.path.join(self.config.output_dir, "embedding_clustering_summary.tsv")
            # Write TSV as a clean table for downstream tools/tests
            summary_df.to_csv(summary_path, sep="\t", index=False)

            # Emit a sidecar file with commented configuration details
            try:
                comments_path = os.path.join(
                    self.config.output_dir, "embedding_clustering_summary.comments.txt"
                )
                header_lines: List[str] = []
                header_lines.append(
                    "# rasembedd clustering summary: run configuration and parameters"
                )
                header_lines.append(
                    "# This file accompanies embedding_clustering_summary.tsv and is informational."
                )
                header_lines.append(f"# embedding_files: {', '.join(self.config.embedding_files)}")
                header_lines.append(f"# metadata_file: {self.config.metadata_file}")
                header_lines.append(f"# output_dir: {self.config.output_dir}")
                header_lines.append(f"# id_column: {self.config.id_column}")
                header_lines.append(f"# label_column: {self.config.label_column}")
                header_lines.append(f"# methods: {', '.join(self.config.methods)}")
                header_lines.append(f"# n_clusters: {self.config.n_clusters}")
                header_lines.append(f"# max_clusters: {self.config.max_clusters}")
                header_lines.append(f"# normalization_method: {self.config.normalization_method}")
                header_lines.append(
                    f"# pipeline: center={self.config.norm_center}, scale={self.config.norm_scale}, pca_components={self.config.norm_pca_components}, l2={self.config.norm_l2}"
                )
                # Add PCA information if available
                if hasattr(self, "_pca_info") and self._pca_info:
                    for emb_name, pca_info in self._pca_info.items():
                        if pca_info:  # Only if PCA was actually applied
                            header_lines.append(
                                f"# pca_info.{emb_name}: actual_components={pca_info.get('pca_n_components', 'N/A')}, "
                                f"variance_explained={pca_info.get('pca_explained_variance_ratio_sum', 0):.4f}, "
                                f"requested={pca_info.get('pca_requested_components', 'N/A')}"
                            )
                header_lines.append(
                    f"# subsample: runs={self.config.subsample}, fraction={self.config.subsample_fraction}, stratified={self.config.stratified_subsample}"
                )
                # Method options
                for method, opts in (self.config.clustering_options or {}).items():
                    header_lines.append(
                        f"# method_options.{method}: {json.dumps(opts, ensure_ascii=False)}"
                    )
                # Per-embedding used params
                for embedding_file, emb_result in embedding_results.items():
                    emb_name = os.path.splitext(os.path.basename(embedding_file))[0]
                    for method, result in emb_result["results"].items():
                        header_lines.append(
                            f"# used_params.{emb_name}.{method}: {json.dumps(result.params, ensure_ascii=False)}"
                        )
                with open(comments_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(header_lines) + "\n")
                print(f"Successfully wrote comments file: {comments_path}")
            except Exception as e:
                # Don't fail saving results if comments sidecar cannot be written
                print(f"Warning: Failed to write comments file {comments_path}: {e}")
                pass

        if params_records:
            params_df = pd.DataFrame(params_records)
            params_df.to_csv(
                os.path.join(self.config.output_dir, "embedding_clustering_parameters.tsv"),
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
        choices=["kmeans", "hierarchical", "dbscan", "hdbscan", "spectral"],
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
        "--k-selection-metric",
        choices=["silhouette", "calinski_harabasz", "davies_bouldin", "elbow"],
        default="silhouette",
        help="Internal metric to use for selecting optimal k (avoids using ground truth)",
    )
    parser.add_argument(
        "--normalization-method",
        choices=["standard", "l2", "pca", "zca", "none", "pipeline"],
        default="l2",
        help="Normalization method (legacy) or 'pipeline' to use the 3-step pipeline.",
    )
    # Pipeline normalization options
    parser.add_argument(
        "--norm-center",
        action="store_true",
        help="Pipeline step 1: apply mean-centering (StandardScaler with_mean=True)",
    )
    parser.add_argument(
        "--norm-no-center",
        action="store_true",
        help="Explicitly disable centering in pipeline (overrides --norm-center)",
    )
    parser.add_argument(
        "--norm-scale",
        action="store_true",
        help="Pipeline step 1: scale to unit variance (StandardScaler with_std=True)",
    )
    parser.add_argument(
        "--norm-no-scale",
        action="store_true",
        help="Explicitly disable scaling in pipeline (overrides --norm-scale)",
    )
    parser.add_argument(
        "--norm-pca-components",
        type=float,
        default=0,
        help="Pipeline step 2: PCA reduction. Int (#components) or float in (0,1] for variance retained. 0 disables.",
    )
    parser.add_argument(
        "--norm-l2",
        action="store_true",
        help="Pipeline step 3: apply L2 normalization (enabled by default unless --norm-no-l2 is used)",
    )
    parser.add_argument(
        "--norm-no-l2",
        action="store_true",
        help="Disable L2 normalization at the end of pipeline",
    )

    # Clustering algorithm options with auto/default/value support
    # K-means parameters
    parser.add_argument(
        "--kmeans-init",
        default="auto",
        help="K-means initialization method. Options: 'auto' (optimized), 'default' (k-means++), 'k-means++', 'random' (default: auto)",
    )
    parser.add_argument(
        "--kmeans-max-iter",
        default="auto",
        help="Maximum iterations for K-means. Options: 'auto' (optimized), 'default' (300), or integer value (default: auto)",
    )

    # Hierarchical clustering parameters
    parser.add_argument(
        "--hierarchical-linkage",
        default="auto",
        help="Linkage criterion for hierarchical clustering. Options: 'auto' (optimized), 'default' (complete), 'ward', 'complete', 'average', 'single' (default: auto)",
    )
    parser.add_argument(
        "--hierarchical-metric",
        default="auto",
        help="Distance metric for hierarchical clustering. Options: 'auto' (optimized), 'default' (euclidean), 'euclidean', 'l1', 'l2', 'manhattan', 'cosine'. Note: ward linkage only supports euclidean (default: auto)",
    )

    # DBSCAN parameters
    parser.add_argument(
        "--dbscan-eps",
        default="auto",
        help="DBSCAN eps parameter. Options: 'auto' (optimized), 'default' (0.5), or float value (default: auto)",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        default="auto",
        help="DBSCAN min_samples parameter. Options: 'auto' (optimized), 'default' (5), or integer value (default: auto)",
    )

    # HDBSCAN parameters
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        default="auto",
        help="HDBSCAN min_cluster_size parameter. Options: 'auto' (optimized), 'default' (5), or integer value (default: auto)",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        default="auto",
        help="HDBSCAN min_samples parameter. Options: 'auto' (optimized), 'default' (None), or integer value (default: auto)",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        default="auto",
        help="HDBSCAN cluster_selection_epsilon parameter. Options: 'auto' (optimized), 'default' (0.0), or float value (default: auto)",
    )

    # Spectral Clustering parameters
    parser.add_argument(
        "--spectral-affinity",
        default="auto",
        help="Spectral Clustering affinity. Options: 'auto' (optimized), 'default' (rbf), 'rbf', 'nearest_neighbors', 'precomputed', 'precomputed_nearest_neighbors' (default: auto)",
    )
    parser.add_argument(
        "--spectral-assign-labels",
        default="auto",
        help="Spectral Clustering label assignment strategy. Options: 'auto' (optimized), 'default' (kmeans), 'kmeans', 'discretize' (default: auto)",
    )
    parser.add_argument(
        "--spectral-n-neighbors",
        default="auto",
        help="Number of neighbors for nearest_neighbors affinity. Options: 'auto' (optimized), 'default' (10), or integer value (default: auto)",
    )
    parser.add_argument(
        "--spectral-gamma",
        default="auto",
        help="Kernel coefficient for rbf affinity. Options: 'auto' (optimized), 'default' (1.0), or float value (default: auto)",
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

    # Helper function to parse parameter values with auto/default/value support
    def parse_clustering_param(value, param_type, default_value, valid_choices=None):
        """Parse clustering parameter with auto/default/value support.

        Args:
            value: The string value from CLI
            param_type: Type to convert to (int, float, str)
            default_value: The sklearn default value
            valid_choices: List of valid string choices (for string parameters)

        Returns:
            Tuple of (parsed_value, is_auto_mode)
            - parsed_value: The actual value to use
            - is_auto_mode: True if optimization should be used
        """
        if value == "auto":
            return default_value, True  # Use default for optimization, but mark as auto
        elif value == "default":
            return default_value, False  # Use default, no optimization
        else:
            # Parse specific value
            if param_type == int:
                try:
                    return int(value), False
                except ValueError:
                    raise ValueError(f"Invalid integer value: {value}")
            elif param_type == float:
                try:
                    return float(value), False
                except ValueError:
                    raise ValueError(f"Invalid float value: {value}")
            elif param_type == str:
                if valid_choices and value not in valid_choices:
                    raise ValueError(f"Invalid choice '{value}'. Valid choices: {valid_choices}")
                return value, False
            else:
                return value, False

    # Parse all clustering parameters
    kmeans_params = {}
    hierarchical_params = {}
    dbscan_params = {}
    hdbscan_params = {}
    spectral_params = {}

    # Track which parameters should be optimized (auto mode)
    auto_params = {"kmeans": {}, "hierarchical": {}, "dbscan": {}, "hdbscan": {}, "spectral": {}}

    # K-means parameters
    kmeans_init, auto_init = parse_clustering_param(
        args.kmeans_init, str, "k-means++", ["k-means++", "random"]
    )
    kmeans_params["init"] = kmeans_init
    if auto_init:
        auto_params["kmeans"]["init"] = True

    kmeans_max_iter, auto_max_iter = parse_clustering_param(args.kmeans_max_iter, int, 300)
    kmeans_params["max_iter"] = kmeans_max_iter
    if auto_max_iter:
        auto_params["kmeans"]["max_iter"] = True

    # Hierarchical parameters
    hier_linkage, auto_linkage = parse_clustering_param(
        args.hierarchical_linkage, str, "complete", ["ward", "complete", "average", "single"]
    )
    hierarchical_params["linkage"] = hier_linkage
    if auto_linkage:
        auto_params["hierarchical"]["linkage"] = True

    hier_metric, auto_metric = parse_clustering_param(
        args.hierarchical_metric, str, "euclidean", ["euclidean", "l1", "l2", "manhattan", "cosine"]
    )
    hierarchical_params["metric"] = hier_metric
    if auto_metric:
        auto_params["hierarchical"]["metric"] = True

    # DBSCAN parameters
    dbscan_eps, auto_eps = parse_clustering_param(args.dbscan_eps, float, 0.5)
    dbscan_params["eps"] = dbscan_eps
    if auto_eps:
        auto_params["dbscan"]["eps"] = True

    dbscan_min_samples, auto_min_samples = parse_clustering_param(args.dbscan_min_samples, int, 5)
    dbscan_params["min_samples"] = dbscan_min_samples
    if auto_min_samples:
        auto_params["dbscan"]["min_samples"] = True

    # HDBSCAN parameters
    hdbscan_min_cluster_size, auto_min_cluster_size = parse_clustering_param(
        args.hdbscan_min_cluster_size, int, 5
    )
    hdbscan_params["min_cluster_size"] = hdbscan_min_cluster_size
    if auto_min_cluster_size:
        auto_params["hdbscan"]["min_cluster_size"] = True

    # HDBSCAN min_samples can be None (special case)
    if args.hdbscan_min_samples == "auto":
        hdbscan_params["min_samples"] = None
        auto_params["hdbscan"]["min_samples"] = True
    elif args.hdbscan_min_samples == "default":
        hdbscan_params["min_samples"] = None
    else:
        hdbscan_params["min_samples"] = (
            int(args.hdbscan_min_samples) if args.hdbscan_min_samples != "None" else None
        )

    hdbscan_epsilon, auto_epsilon = parse_clustering_param(
        args.hdbscan_cluster_selection_epsilon, float, 0.0
    )
    hdbscan_params["cluster_selection_epsilon"] = hdbscan_epsilon
    if auto_epsilon:
        auto_params["hdbscan"]["cluster_selection_epsilon"] = True

    # Spectral parameters
    spectral_affinity, auto_affinity = parse_clustering_param(
        args.spectral_affinity,
        str,
        "rbf",
        ["rbf", "nearest_neighbors", "precomputed", "precomputed_nearest_neighbors"],
    )
    spectral_params["affinity"] = spectral_affinity
    if auto_affinity:
        auto_params["spectral"]["affinity"] = True

    spectral_assign_labels, auto_assign_labels = parse_clustering_param(
        args.spectral_assign_labels, str, "kmeans", ["kmeans", "discretize"]
    )
    spectral_params["assign_labels"] = spectral_assign_labels
    if auto_assign_labels:
        auto_params["spectral"]["assign_labels"] = True

    spectral_n_neighbors, auto_n_neighbors = parse_clustering_param(
        args.spectral_n_neighbors, int, 10
    )
    spectral_params["n_neighbors"] = spectral_n_neighbors
    if auto_n_neighbors:
        auto_params["spectral"]["n_neighbors"] = True

    # Spectral gamma parameter (sklearn default is 1.0 for RBF kernel)
    if args.spectral_gamma == "auto":
        spectral_params["gamma"] = 1.0  # Use sklearn default for optimization
        auto_params["spectral"]["gamma"] = True
    elif args.spectral_gamma == "default":
        spectral_params["gamma"] = 1.0  # sklearn default
    else:
        try:
            spectral_params["gamma"] = float(args.spectral_gamma)
        except ValueError:
            raise ValueError(f"Invalid gamma value: {args.spectral_gamma}")

    # Build clustering options dictionary with auto parameter information
    clustering_options = {
        "kmeans": kmeans_params,
        "hierarchical": hierarchical_params,
        "dbscan": dbscan_params,
        "hdbscan": hdbscan_params,
        "spectral": spectral_params,
    }

    # Store auto parameter information with special keys
    clustering_options["_auto_params"] = auto_params

    # Validate hierarchical clustering parameters
    if args.hierarchical_linkage == "ward" and args.hierarchical_metric != "euclidean":
        print("Warning: Ward linkage only supports euclidean metric.")
        print(
            f"You specified --hierarchical-metric {args.hierarchical_metric} with --hierarchical-linkage ward."
        )
        print(
            "Please use --hierarchical-linkage complete (or average/single) with manhattan metric."
        )
        print("Or use --hierarchical-metric euclidean with ward linkage.")
        raise ValueError(
            "Invalid linkage-metric combination: ward linkage requires euclidean metric"
        )

    return ClusteringConfig(
        embedding_files=args.embedding_files,
        metadata_file=args.metadata_file,
        output_dir=args.output_dir,
        id_column=args.id_column,
        label_column=args.label_column,
        methods=args.methods,
        n_clusters=args.n_clusters,
        max_clusters=args.max_clusters,
        k_selection_metric=args.k_selection_metric,
        normalization_method=args.normalization_method,
        norm_center=(True if args.norm_center else False) if not args.norm_no_center else False,
        norm_scale=(True if args.norm_scale else False) if not args.norm_no_scale else False,
        norm_pca_components=args.norm_pca_components,
        norm_l2=False if args.norm_no_l2 else True,
        clustering_options=clustering_options,
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
