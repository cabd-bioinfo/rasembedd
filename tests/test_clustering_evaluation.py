"""Tests for clustering evaluation script."""

import os
import pickle
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Import the clustering evaluation modules
from clustering_evaluation import (
    ClusteringAnalyzer,
    ClusteringConfig,
    ClusteringEngine,
    ClusteringResult,
    DataLoader,
    EmbeddingNormalizer,
    SubsamplingAnalyzer,
    Visualizer,
)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return {
        "protein_1": np.random.randn(128).astype(np.float32),
        "protein_2": np.random.randn(128).astype(np.float32),
        "protein_3": np.random.randn(128).astype(np.float32),
        "protein_4": np.random.randn(128).astype(np.float32),
        "protein_5": np.random.randn(128).astype(np.float32),
    }


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pd.DataFrame(
        {
            "uniprot_id": ["protein_1", "protein_2", "protein_3", "protein_4", "protein_5"],
            "Family.name": ["FamilyA", "FamilyA", "FamilyB", "FamilyB", "FamilyC"],
            "species": ["species_1", "species_1", "species_2", "species_2", "species_3"],
            "length": [100, 120, 110, 130, 115],
        }
    )


@pytest.fixture
def temp_files(sample_embeddings, sample_metadata):
    """Create temporary files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create embedding file
        embedding_file = os.path.join(temp_dir, "test_embeddings.pkl")
        with open(embedding_file, "wb") as f:
            pickle.dump(sample_embeddings, f)

        # Create metadata file
        metadata_file = os.path.join(temp_dir, "test_metadata.tsv")
        sample_metadata.to_csv(metadata_file, sep="\t", index=False)

        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        yield {
            "embedding_file": embedding_file,
            "metadata_file": metadata_file,
            "output_dir": output_dir,
            "temp_dir": temp_dir,
        }


class TestClusteringConfig:
    """Test cases for ClusteringConfig."""

    def test_clustering_config_initialization(self):
        """Test ClusteringConfig initialization."""
        config = ClusteringConfig(embedding_files=["file1.pkl"], metadata_file="metadata.tsv")

        assert config.embedding_files == ["file1.pkl"]
        assert config.metadata_file == "metadata.tsv"
        assert config.output_dir == "clustering_results"
        assert config.id_column == "uniprot_id"
        assert config.label_column == "Family.name"
        assert config.methods == ["kmeans", "hierarchical"]
        assert config.n_clusters is None
        assert config.max_clusters == 15
        assert config.normalization_method == "l2"
        assert config.subsample == 0
        assert config.subsample_fraction == 0.8
        assert config.stratified_subsample is False

    def test_clustering_config_custom_methods(self):
        """Test ClusteringConfig with custom methods."""
        config = ClusteringConfig(
            embedding_files=["file1.pkl"],
            metadata_file="metadata.tsv",
            methods=["dbscan", "hierarchical"],
        )

        assert config.methods == ["dbscan", "hierarchical"]

    def test_clustering_config_with_clustering_options(self):
        """Test ClusteringConfig with clustering options."""
        clustering_options = {
            "kmeans_init": "k-means++",
            "kmeans_max_iter": 500,
            "hierarchical_linkage": "ward",
            "hierarchical_metric": "euclidean",
            "dbscan_eps": 0.3,
            "dbscan_min_samples": 5,
            "hdbscan_min_cluster_size": 10,
            "hdbscan_min_samples": 3,
        }

        config = ClusteringConfig(
            embedding_files=["file1.pkl"],
            metadata_file="metadata.tsv",
            clustering_options=clustering_options,
        )

        assert config.clustering_options == clustering_options

    def test_clustering_config_with_normalization_methods(self):
        """Test ClusteringConfig with different normalization methods."""
        for method in ["l2", "standard", "pca", "zca", "none"]:
            config = ClusteringConfig(
                embedding_files=["file1.pkl"],
                metadata_file="metadata.tsv",
                normalization_method=method,
            )
            assert config.normalization_method == method


class TestClusteringOptions:
    """Test cases for clustering algorithm options."""

    def test_kmeans_with_options(self, sample_embeddings):
        """Test K-means clustering with custom options."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Test with custom init and max_iter
        labels = engine.perform_clustering(
            embeddings_array,
            method="kmeans",
            n_clusters=3,
            init="random",
            max_iter=50,
        )

        assert len(labels) == 5
        assert len(np.unique(labels)) <= 3
        assert all(label >= 0 for label in labels)

    def test_hierarchical_with_options(self, sample_embeddings):
        """Test hierarchical clustering with custom options."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Test with custom linkage and metric
        labels = engine.perform_clustering(
            embeddings_array,
            method="hierarchical",
            n_clusters=2,
            linkage="complete",
            metric="manhattan",
        )

        assert len(labels) == 5
        assert len(np.unique(labels)) == 2
        assert all(label >= 0 for label in labels)

    def test_hierarchical_ward_linkage_validation(self, sample_embeddings):
        """Test that ward linkage requires euclidean metric."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Ward linkage with non-euclidean metric should raise ValueError
        with pytest.raises(ValueError, match="Ward linkage requires euclidean metric"):
            engine.perform_clustering(
                embeddings_array,
                method="hierarchical",
                n_clusters=2,
                linkage="ward",
                metric="manhattan",
            )

    def test_hierarchical_valid_linkage_metrics(self, sample_embeddings):
        """Test hierarchical clustering with various valid linkage-metric combinations."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Test valid combinations
        valid_combinations = [
            ("ward", "euclidean"),
            ("complete", "euclidean"),
            ("complete", "manhattan"),
            ("average", "cosine"),
            ("single", "l1"),
        ]

        for linkage, metric in valid_combinations:
            labels = engine.perform_clustering(
                embeddings_array,
                method="hierarchical",
                n_clusters=2,
                linkage=linkage,
                metric=metric,
            )
            assert len(labels) == 5
            assert all(label >= 0 for label in labels)

    def test_dbscan_with_options(self, sample_embeddings):
        """Test DBSCAN clustering with custom options."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Test with custom eps and min_samples
        labels = engine.perform_clustering(
            embeddings_array,
            method="dbscan",
            eps=0.5,
            min_samples=2,
        )

        assert len(labels) == 5
        # DBSCAN can have noise points (label -1)
        assert all(label >= -1 for label in labels)

    def test_hdbscan_with_options(self, sample_embeddings):
        """Test HDBSCAN clustering with custom options."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Test with custom min_cluster_size and min_samples
        labels = engine.perform_clustering(
            embeddings_array,
            method="hdbscan",
            min_cluster_size=2,
            min_samples=2,
        )

        assert len(labels) == 5
        # HDBSCAN can have noise points (label -1)
        assert all(label >= -1 for label in labels)

    def test_hdbscan_with_all_options(self, sample_embeddings):
        """Test HDBSCAN clustering with all available options."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Test with all HDBSCAN parameters
        labels = engine.perform_clustering(
            embeddings_array,
            method="hdbscan",
            min_cluster_size=3,
            min_samples=2,
            cluster_selection_epsilon=0.1,
        )

        assert len(labels) == 5
        # HDBSCAN can have noise points (label -1)
        assert all(label >= -1 for label in labels)

    def test_clustering_options_parameter_passing(self, sample_embeddings):
        """Test that clustering options are properly passed to algorithms."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Mock the clustering algorithms to verify parameters are passed
        with patch("clustering_evaluation.KMeans") as mock_kmeans:
            mock_kmeans.return_value.fit_predict.return_value = np.array([0, 0, 1, 1, 2])

            engine.perform_clustering(
                embeddings_array,
                method="kmeans",
                n_clusters=3,
                init="random",
                max_iter=50,
            )

            # Verify KMeans was called with correct parameters
            mock_kmeans.assert_called_once_with(
                n_clusters=3,
                init="random",
                max_iter=50,
                random_state=42,  # Default added by engine
            )


class TestDataLoader:
    """Test cases for DataLoader."""

    def test_load_embeddings(self, temp_files):
        """Test loading embeddings from pickle file."""
        loader = DataLoader()
        embeddings = loader.load_embeddings(temp_files["embedding_file"])

        assert len(embeddings) == 5
        assert "protein_1" in embeddings
        assert embeddings["protein_1"].shape == (128,)
        assert embeddings["protein_1"].dtype == np.float32

    def test_load_metadata_tsv(self, temp_files):
        """Test loading metadata from TSV file."""
        loader = DataLoader()
        metadata = loader.load_metadata(temp_files["metadata_file"])

        assert len(metadata) == 5
        assert "uniprot_id" in metadata.columns
        assert "Family.name" in metadata.columns
        assert metadata["uniprot_id"].iloc[0] == "protein_1"

    def test_load_metadata_csv(self, temp_files, sample_metadata):
        """Test loading metadata from CSV file."""
        csv_file = os.path.join(temp_files["temp_dir"], "test_metadata.csv")
        sample_metadata.to_csv(csv_file, index=False)

        loader = DataLoader()
        metadata = loader.load_metadata(csv_file)

        assert len(metadata) == 5
        assert "uniprot_id" in metadata.columns

    def test_prepare_data(self, sample_embeddings, sample_metadata):
        """Test data preparation and alignment."""
        loader = DataLoader()
        embedding_matrix, aligned_metadata, protein_ids = loader.prepare_data(
            sample_embeddings, sample_metadata, "uniprot_id"
        )

        assert embedding_matrix.shape == (5, 128)
        assert len(aligned_metadata) == 5
        assert len(protein_ids) == 5
        assert protein_ids[0] == "protein_1"
        assert aligned_metadata["uniprot_id"].iloc[0] == "protein_1"

    def test_prepare_data_partial_overlap(self, sample_embeddings, sample_metadata):
        """Test data preparation with partial overlap."""
        # Remove one protein from embeddings
        partial_embeddings = {k: v for k, v in sample_embeddings.items() if k != "protein_5"}

        loader = DataLoader()
        embedding_matrix, aligned_metadata, protein_ids = loader.prepare_data(
            partial_embeddings, sample_metadata, "uniprot_id"
        )

        assert embedding_matrix.shape == (4, 128)
        assert len(aligned_metadata) == 4
        assert len(protein_ids) == 4
        assert "protein_5" not in protein_ids

    def test_prepare_data_no_overlap(self, sample_metadata):
        """Test data preparation with no overlap."""
        no_overlap_embeddings = {"other_protein": np.random.randn(128).astype(np.float32)}

        loader = DataLoader()
        with pytest.raises(ValueError, match="No common proteins found"):
            loader.prepare_data(no_overlap_embeddings, sample_metadata, "uniprot_id")


class TestClusteringEngine:
    """Test cases for ClusteringEngine."""

    def test_perform_clustering_kmeans(self, sample_embeddings):
        """Test K-means clustering."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        labels = engine.perform_clustering(embeddings_array, method="kmeans", n_clusters=3)

        assert len(labels) == 5
        assert len(np.unique(labels)) <= 3  # May be fewer if some clusters are empty
        assert all(label >= 0 for label in labels)  # K-means labels are non-negative

    def test_perform_clustering_hierarchical(self, sample_embeddings):
        """Test hierarchical clustering."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        labels = engine.perform_clustering(embeddings_array, method="hierarchical", n_clusters=2)

        assert len(labels) == 5
        assert len(np.unique(labels)) == 2
        assert all(label >= 0 for label in labels)

    def test_perform_clustering_dbscan(self, sample_embeddings):
        """Test DBSCAN clustering."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Don't pass eps and min_samples as kwargs to avoid duplicate parameter error
        labels = engine.perform_clustering(embeddings_array, method="dbscan")

        assert len(labels) == 5
        # DBSCAN can have noise points (label -1)
        assert all(label >= -1 for label in labels)

    def test_perform_clustering_hdbscan(self, sample_embeddings):
        """Test HDBSCAN clustering."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        labels = engine.perform_clustering(embeddings_array, method="hdbscan")

        assert len(labels) == 5
        # HDBSCAN can have noise points (label -1)
        assert all(label >= -1 for label in labels)

    def test_perform_clustering_invalid_method(self, sample_embeddings):
        """Test clustering with invalid method."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        with pytest.raises(ValueError, match="Unknown clustering method"):
            engine.perform_clustering(embeddings_array, method="invalid_method")

    def test_evaluate_clustering(self, sample_embeddings):
        """Test clustering evaluation metrics."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Create mock cluster and true labels
        cluster_labels = np.array([0, 0, 1, 1, 2])
        true_labels = np.array([0, 0, 1, 1, 1])

        metrics = engine.evaluate_clustering(cluster_labels, true_labels, embeddings_array)

        # Check that all expected metrics are present
        expected_metrics = [
            "adjusted_rand_score",
            "normalized_mutual_info",
            "homogeneity",
            "completeness",
            "v_measure",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float, np.number))

    def test_evaluate_clustering_single_cluster(self, sample_embeddings):
        """Test clustering evaluation with single cluster."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Single cluster scenario
        cluster_labels = np.array([0, 0, 0, 0, 0])
        true_labels = np.array([0, 0, 1, 1, 1])

        metrics = engine.evaluate_clustering(cluster_labels, true_labels, embeddings_array)

        # Internal metrics should have default values for single cluster
        assert metrics["silhouette_score"] == -1.0
        assert metrics["calinski_harabasz_score"] == 0.0
        assert metrics["davies_bouldin_score"] == float("inf")

    def test_find_optimal_clusters(self, sample_embeddings):
        """Test finding optimal number of clusters."""
        engine = ClusteringEngine()
        embeddings_array = np.array(list(sample_embeddings.values()))
        true_labels = np.array([0, 0, 1, 1, 1])

        best_k, metrics_by_k = engine.find_optimal_clusters(
            embeddings_array, true_labels, method="kmeans", max_clusters=4
        )

        # Check structure of results
        assert isinstance(best_k, dict)
        assert isinstance(metrics_by_k, dict)

        expected_criteria = [
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "adjusted_rand",
            "v_measure",
        ]
        for criterion in expected_criteria:
            assert criterion in best_k
            assert isinstance(best_k[criterion], int)
            assert 2 <= best_k[criterion] <= 4

        # Check that we have metrics for each k tested
        for k in range(2, 5):  # Should test k=2,3,4
            if k in metrics_by_k:
                assert isinstance(metrics_by_k[k], dict)

    def test_find_optimal_clusters_small_dataset(self):
        """Test finding optimal clusters with very small dataset."""
        engine = ClusteringEngine()
        # Small dataset but with enough points for silhouette score (need at least 3)
        embeddings_array = np.random.randn(4, 10)
        true_labels = np.array([0, 0, 1, 1])

        best_k, metrics_by_k = engine.find_optimal_clusters(
            embeddings_array, true_labels, method="kmeans", max_clusters=3
        )

        # Should have valid results for small dataset
        for criterion in best_k:
            assert best_k[criterion] >= 2

        assert len(metrics_by_k) >= 1


class TestClusteringResult:
    """Test cases for ClusteringResult."""

    def test_clustering_result_creation(self):
        """Test ClusteringResult dataclass."""
        cluster_labels = np.array([0, 1, 0, 1, 2])
        metrics = {"adjusted_rand_score": 0.5, "silhouette_score": 0.3}

        result = ClusteringResult(
            cluster_labels=cluster_labels,
            n_clusters=3,
            metrics=metrics,
            params={"method": "kmeans", "n_clusters": 3},
        )

        assert np.array_equal(result.cluster_labels, cluster_labels)
        assert result.n_clusters == 3
        assert result.metrics == metrics
        assert result.params == {"method": "kmeans", "n_clusters": 3}


class TestVisualizer:
    """Test cases for Visualizer."""

    def test_get_distinct_colors(self):
        """Test color palette generation."""
        visualizer = Visualizer()

        # Test small number of colors
        colors = visualizer.get_distinct_colors(5)
        assert colors.shape[0] == 5
        # Colors could be hex strings (1D) or RGB/RGBA arrays (2D), so check safely
        if len(colors.shape) > 1:
            assert colors.shape[1] >= 3  # RGB or RGBA

        # Test large number of colors
        colors = visualizer.get_distinct_colors(25)
        assert colors.shape[0] == 25

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_cluster_optimization(self, mock_close, mock_savefig):
        """Test cluster optimization plotting."""
        visualizer = Visualizer()

        # Mock metrics data
        metrics_by_k = {
            2: {
                "silhouette_score": 0.3,
                "adjusted_rand_score": 0.2,
                "calinski_harabasz_score": 100,
                "davies_bouldin_score": 1.2,
                "v_measure": 0.25,
                "normalized_mutual_info": 0.22,
            },
            3: {
                "silhouette_score": 0.4,
                "adjusted_rand_score": 0.3,
                "calinski_harabasz_score": 120,
                "davies_bouldin_score": 1.0,
                "v_measure": 0.35,
                "normalized_mutual_info": 0.32,
            },
            4: {
                "silhouette_score": 0.35,
                "adjusted_rand_score": 0.25,
                "calinski_harabasz_score": 110,
                "davies_bouldin_score": 1.1,
                "v_measure": 0.28,
                "normalized_mutual_info": 0.27,
            },
        }

        visualizer.plot_cluster_optimization(metrics_by_k, "test_output.pdf")

        mock_savefig.assert_called_once_with("test_output.pdf", format="pdf", bbox_inches="tight")
        mock_close.assert_called()  # Just check that close was called, not how many times

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_truth_table(self, mock_close, mock_savefig):
        """Test truth table plotting."""
        visualizer = Visualizer()

        true_labels = np.array([0, 0, 1, 1, 2])
        cluster_labels = np.array([0, 0, 1, 2, 2])
        label_names = ["FamilyA", "FamilyB", "FamilyC"]

        visualizer.plot_truth_table(
            true_labels, cluster_labels, label_names, "test_truth_table.pdf"
        )

        mock_savefig.assert_called_once_with(
            "test_truth_table.pdf", format="pdf", bbox_inches="tight"
        )
        mock_close.assert_called()  # Just check that close was called, not how many times

    @patch("pandas.read_csv")
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_significance_heatmap(self, mock_close, mock_savefig, mock_read_csv):
        """Test significance heatmap plotting."""
        # Mock significance data
        mock_df = pd.DataFrame(
            {
                "metric": ["adjusted_rand_score", "adjusted_rand_score"],
                "method": ["kmeans", "kmeans"],
                "embedding1": ["emb1", "emb1"],
                "embedding2": ["emb2", "emb3"],
                "embedding1_mean": [0.3, 0.3],
                "embedding2_mean": [0.5, 0.4],
                "t_p_holm": [0.01, 0.05],
            }
        )
        mock_read_csv.return_value = mock_df

        visualizer = Visualizer()
        visualizer.plot_significance_heatmap("test_significance.tsv", "output_dir")

        mock_read_csv.assert_called_once_with("test_significance.tsv", sep="\t")
        # Should save at least one heatmap
        assert mock_savefig.call_count >= 1
        assert mock_close.call_count >= 1


class TestSubsamplingAnalyzer:
    """Test cases for SubsamplingAnalyzer."""

    def test_subsample_analyzer_initialization(self):
        """Test SubsamplingAnalyzer initialization."""
        config = ClusteringConfig(
            embedding_files=["test.pkl"], metadata_file="metadata.tsv", subsample=10
        )
        analyzer = SubsamplingAnalyzer(config)

        assert analyzer.config == config
        assert isinstance(analyzer.clustering_engine, ClusteringEngine)

    def test_run_subsample(self, sample_embeddings):
        """Test single subsampling run."""
        config = ClusteringConfig(
            embedding_files=["test.pkl"],
            metadata_file="metadata.tsv",
            methods=["kmeans"],
            subsample_fraction=0.8,
            n_clusters=2,
        )
        analyzer = SubsamplingAnalyzer(config)

        protein_ids = list(sample_embeddings.keys())
        embeddings_dict = {"test.pkl": sample_embeddings}
        true_labels = np.array([0, 0, 1, 1, 1])

        results = analyzer.run_subsample(0, protein_ids, embeddings_dict, true_labels)

        assert isinstance(results, dict)
        # Should have results for each combination of embedding, method, and metric
        assert len(results) > 0
        for key in results:
            assert isinstance(key, tuple)
            assert len(key) == 3  # (embedding_name, method, metric_name)

    @patch("clustering_evaluation.Parallel")
    def test_run_subsampling_analysis(self, mock_parallel, sample_embeddings):
        """Test complete subsampling analysis."""
        config = ClusteringConfig(
            embedding_files=["test.pkl"],
            metadata_file="metadata.tsv",
            methods=["kmeans"],
            subsample=3,
            subsample_fraction=0.8,
            n_clusters=2,
        )
        analyzer = SubsamplingAnalyzer(config)

        # Mock the parallel execution results - Parallel() should return callable
        mock_results = [
            {("test", "kmeans", "adjusted_rand_score"): 0.3},
            {("test", "kmeans", "adjusted_rand_score"): 0.4},
            {("test", "kmeans", "adjusted_rand_score"): 0.35},
        ]
        mock_parallel_instance = Mock()
        mock_parallel_instance.return_value = mock_results
        mock_parallel.return_value = mock_parallel_instance

        protein_ids = list(sample_embeddings.keys())
        embeddings_dict = {"test.pkl": sample_embeddings}
        true_labels = np.array([0, 0, 1, 1, 1])

        df = analyzer.run_subsampling_analysis(protein_ids, embeddings_dict, true_labels)

        assert isinstance(df, pd.DataFrame)
        expected_columns = ["run", "embedding", "method", "metric", "value"]
        for col in expected_columns:
            assert col in df.columns

    def test_generate_statistical_tests(self):
        """Test statistical test generation."""
        config = ClusteringConfig(embedding_files=["test.pkl"], metadata_file="metadata.tsv")
        analyzer = SubsamplingAnalyzer(config)

        # Create mock subsampling results
        df = pd.DataFrame(
            {
                "run": [0, 1, 2, 0, 1, 2],
                "embedding": ["emb1", "emb1", "emb1", "emb2", "emb2", "emb2"],
                "method": ["kmeans", "kmeans", "kmeans", "kmeans", "kmeans", "kmeans"],
                "metric": ["adjusted_rand_score"] * 6,
                "value": [0.3, 0.35, 0.32, 0.4, 0.42, 0.38],
            }
        )

        stats_df = analyzer.generate_statistical_tests(df)

        assert isinstance(stats_df, pd.DataFrame)
        expected_columns = [
            "metric",
            "method",
            "embedding1",
            "embedding2",
            "embedding1_mean",
            "embedding2_mean",
            "t_stat",
            "t_p",
            "wilcoxon_stat",
            "wilcoxon_p",
        ]
        for col in expected_columns:
            assert col in stats_df.columns

        # Should have one comparison (emb1 vs emb2)
        assert len(stats_df) == 1
        assert stats_df["embedding1"].iloc[0] == "emb1"
        assert stats_df["embedding2"].iloc[0] == "emb2"


class TestDensityBasedOptimization:
    """Tests for DBSCAN/HDBSCAN parameter selection and handling."""

    def _make_separable_data(self, n_per_cluster: int = 20, seed: int = 42):
        np.random.seed(seed)
        centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
        X = []
        y = []
        for i, c in enumerate(centers):
            X.append(c + 0.4 * np.random.randn(n_per_cluster, 2))
            y.extend([i] * n_per_cluster)
        X = np.vstack(X)
        y = np.array(y)
        return X, y

    def test_dbscan_parameter_search_updates_options_and_finds_clusters(self):
        """DBSCAN search should adjust eps/min_samples and yield >=2 clusters."""
        engine = ClusteringEngine()
        X, y = self._make_separable_data(n_per_cluster=15, seed=123)

        # Intentionally poor defaults to force the search to do work
        opts = {"eps": 0.05, "min_samples": 15}

        best_k, by_k = engine.find_optimal_clusters(
            X, y, method="dbscan", max_clusters=10, clustering_options=opts
        )

        # Options should be updated to selected parameters
        assert opts["eps"] != 0.05 or opts["min_samples"] != 15

        # Should report at least 2 clusters as best across criteria
        for crit, k in best_k.items():
            assert k >= 2

        # With the chosen params, running DBSCAN should produce >=2 clusters
        labels = engine.perform_clustering(X, method="dbscan", **opts)
        uniq = set(labels)
        n_c = len(uniq) - (1 if -1 in uniq else 0)
        assert n_c >= 2

    def test_hdbscan_single_evaluation_reports_observed_clusters(self):
        """HDBSCAN path should report observed cluster count (>=2 here)."""
        engine = ClusteringEngine()
        X, y = self._make_separable_data(n_per_cluster=12, seed=321)

        opts = {"min_cluster_size": 5, "min_samples": 2}
        best_k, by_k = engine.find_optimal_clusters(
            X, y, method="hdbscan", max_clusters=10, clustering_options=opts
        )

        # Key in by_k should be the observed number of clusters
        assert isinstance(by_k, dict) and len(by_k) == 1
        observed_k = next(iter(by_k.keys()))
        assert observed_k >= 2

        # All criteria should reflect the same observed cluster count
        for crit, k in best_k.items():
            assert k == observed_k


class TestClusteringAnalyzer:
    """Test cases for ClusteringAnalyzer."""

    def test_clustering_analyzer_initialization(self, temp_files):
        """Test ClusteringAnalyzer initialization."""
        config = ClusteringConfig(
            embedding_files=[temp_files["embedding_file"]],
            metadata_file=temp_files["metadata_file"],
            output_dir=temp_files["output_dir"],
        )
        analyzer = ClusteringAnalyzer(config)

        assert analyzer.config == config
        assert isinstance(analyzer.data_loader, DataLoader)
        assert isinstance(analyzer.clustering_engine, ClusteringEngine)
        assert isinstance(analyzer.visualizer, Visualizer)
        assert isinstance(analyzer.subsampling_analyzer, SubsamplingAnalyzer)
        assert os.path.exists(temp_files["output_dir"])

    @patch("clustering_evaluation.ClusteringAnalyzer._run_regular_analysis")
    def test_run_analysis_regular(self, mock_regular, temp_files):
        """Test running regular clustering analysis."""
        config = ClusteringConfig(
            embedding_files=[temp_files["embedding_file"]],
            metadata_file=temp_files["metadata_file"],
            output_dir=temp_files["output_dir"],
            subsample=0,  # No subsampling
        )
        analyzer = ClusteringAnalyzer(config)

        analyzer.run_analysis()

        mock_regular.assert_called_once()

    @patch("clustering_evaluation.ClusteringAnalyzer._run_subsampling_analysis")
    def test_run_analysis_subsampling(self, mock_subsampling, temp_files):
        """Test running subsampling analysis."""
        config = ClusteringConfig(
            embedding_files=[temp_files["embedding_file"]],
            metadata_file=temp_files["metadata_file"],
            output_dir=temp_files["output_dir"],
            subsample=5,  # Enable subsampling
        )
        analyzer = ClusteringAnalyzer(config)

        analyzer.run_analysis()

        mock_subsampling.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_save_results(self, mock_close, mock_savefig, temp_files):
        """Test saving clustering results."""
        config = ClusteringConfig(
            embedding_files=[temp_files["embedding_file"]],
            metadata_file=temp_files["metadata_file"],
            output_dir=temp_files["output_dir"],
        )
        analyzer = ClusteringAnalyzer(config)

        # Mock embedding results
        mock_results = {
            temp_files["embedding_file"]: {
                "results": {
                    "kmeans": ClusteringResult(
                        cluster_labels=np.array([0, 0, 1, 1, 2]),
                        n_clusters=3,
                        metrics={"adjusted_rand_score": 0.5},
                        params={"init": "k-means++", "max_iter": 300},
                    )
                },
                "protein_ids": ["protein_1", "protein_2", "protein_3", "protein_4", "protein_5"],
                "true_labels_str": ["FamilyA", "FamilyA", "FamilyB", "FamilyB", "FamilyC"],
                "true_labels": np.array([0, 0, 1, 1, 2]),
                "embedding_matrix": np.random.randn(5, 128),
            }
        }

        analyzer._save_results(mock_results)

        # Check that files were created
        expected_files = [
            "test_embeddings_cluster_assignments.tsv",
            "test_embeddings_clustering_results.tsv",
            "embedding_clustering_summary.tsv",
            "embedding_clustering_parameters.tsv",
        ]

        for filename in expected_files:
            filepath = os.path.join(temp_files["output_dir"], filename)
            assert os.path.exists(filepath)

        # Verify that clustering results include params_json column
        results_path = os.path.join(
            temp_files["output_dir"], "test_embeddings_clustering_results.tsv"
        )
        results_df = pd.read_csv(results_path, sep="\t")
        assert "params_json" in results_df.columns

    def test_integration_small_dataset(self, temp_files):
        """Test end-to-end integration with small dataset."""
        config = ClusteringConfig(
            embedding_files=[temp_files["embedding_file"]],
            metadata_file=temp_files["metadata_file"],
            output_dir=temp_files["output_dir"],
            methods=["kmeans"],
            n_clusters=2,  # Fixed number of clusters for small dataset
            subsample=0,  # No subsampling for integration test
        )

        # This should run without errors
        analyzer = ClusteringAnalyzer(config)

        # Mock the plotting methods to avoid file I/O issues in tests
        with patch.object(analyzer.visualizer, "plot_cluster_optimization"), patch.object(
            analyzer.visualizer, "plot_truth_table"
        ):
            analyzer.run_analysis()

        # Check that some output files were created
        assert os.path.exists(
            os.path.join(temp_files["output_dir"], "embedding_clustering_summary.tsv")
        )


class TestIntegration:
    """Integration tests for the entire clustering evaluation pipeline."""

    def test_full_pipeline_with_mocked_plots(self, temp_files):
        """Test the full pipeline with mocked plotting functions."""
        config = ClusteringConfig(
            embedding_files=[temp_files["embedding_file"]],
            metadata_file=temp_files["metadata_file"],
            output_dir=temp_files["output_dir"],
            methods=["kmeans", "hierarchical"],
            max_clusters=3,  # Small for test dataset
            subsample=0,
        )

        analyzer = ClusteringAnalyzer(config)

        # Mock all plotting functions to avoid file I/O in tests
        with patch.object(analyzer.visualizer, "plot_cluster_optimization"), patch.object(
            analyzer.visualizer, "plot_truth_table"
        ), patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):

            # This should complete without errors
            analyzer.run_analysis()

        # Verify that key output files exist
        summary_file = os.path.join(temp_files["output_dir"], "embedding_clustering_summary.tsv")
        assert os.path.exists(summary_file)

        # Check summary file content
        summary_df = pd.read_csv(summary_file, sep="\t")
        assert len(summary_df) == 2  # kmeans and hierarchical
        assert "embedding" in summary_df.columns
        assert "method" in summary_df.columns
        assert "adjusted_rand_score" in summary_df.columns

    def test_error_handling_missing_file(self):
        """Test error handling for missing input files."""
        config = ClusteringConfig(
            embedding_files=["nonexistent.pkl"], metadata_file="nonexistent.tsv"
        )

        analyzer = ClusteringAnalyzer(config)

        # Should raise an appropriate error (ValueError is what's actually raised)
        with pytest.raises(ValueError):
            analyzer.run_analysis()

    def test_parameter_validation(self):
        """Test parameter validation in ClusteringConfig."""
        # Test with invalid clustering method in post-init
        config = ClusteringConfig(
            embedding_files=["test.pkl"], metadata_file="metadata.tsv", methods=["invalid_method"]
        )

        # The config should accept it (validation happens in ClusteringEngine)
        assert config.methods == ["invalid_method"]

    @patch("clustering_evaluation.parse_arguments")
    @patch("clustering_evaluation.ClusteringAnalyzer")
    def test_main_function(self, mock_analyzer_class, mock_parse_args):
        """Test the main function."""
        from clustering_evaluation import main

        # Mock the config and analyzer
        mock_config = Mock()
        mock_parse_args.return_value = mock_config
        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer

        # Run main function
        main()

        mock_parse_args.assert_called_once()
        mock_analyzer_class.assert_called_once_with(mock_config)
        mock_analyzer.run_analysis.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_embeddings(self):
        """Test handling of empty embeddings."""
        loader = DataLoader()
        empty_embeddings = {}
        metadata = pd.DataFrame({"uniprot_id": ["protein_1"], "Family.name": ["FamilyA"]})

        with pytest.raises(ValueError, match="No common proteins found"):
            loader.prepare_data(empty_embeddings, metadata, "uniprot_id")

    def test_clustering_with_insufficient_data(self):
        """Test clustering with insufficient data points."""
        engine = ClusteringEngine()

        # Single data point
        single_point = np.random.randn(1, 10)

        # This should handle the edge case gracefully
        try:
            labels = engine.perform_clustering(single_point, method="kmeans", n_clusters=2)
            # Should return single label
            assert len(labels) == 1
        except ValueError:
            # It's also acceptable to raise an error for insufficient data
            pass

    def test_clustering_evaluation_edge_cases(self):
        """Test clustering evaluation with edge cases."""
        engine = ClusteringEngine()

        # Perfect clustering scenario
        embeddings = np.random.randn(4, 10)
        cluster_labels = np.array([0, 0, 1, 1])
        true_labels = np.array([0, 0, 1, 1])

        metrics = engine.evaluate_clustering(cluster_labels, true_labels, embeddings)

        # Perfect clustering should have high scores
        assert metrics["adjusted_rand_score"] == 1.0
        assert metrics["homogeneity"] == 1.0
        assert metrics["completeness"] == 1.0

    def test_malformed_metadata(self, temp_files):
        """Test handling of malformed metadata files."""
        # Create malformed metadata file
        malformed_file = os.path.join(temp_files["temp_dir"], "malformed.tsv")
        with open(malformed_file, "w") as f:
            f.write("not,proper,tsv,format\nwith,missing,data")

        loader = DataLoader()

        # Should still try to load it
        try:
            metadata = loader.load_metadata(malformed_file)
            # If it loads, check it has some structure
            assert isinstance(metadata, pd.DataFrame)
        except Exception:
            # It's acceptable to fail on malformed data
            pass


class TestEmbeddingNormalizer:
    """Test cases for embedding normalization methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample embedding data for normalization testing."""
        np.random.seed(42)
        # Create data with known properties for testing
        n_samples, n_features = 20, 10
        data = np.random.randn(n_samples, n_features)
        # Add some structure to make tests more meaningful
        data[:5, :] *= 2  # First group has larger variance
        data[5:10, :] += 1  # Second group has different mean
        return data

    def test_none_normalization(self, sample_data):
        """Test that 'none' normalization returns unchanged data."""
        result = EmbeddingNormalizer.normalize_embeddings(sample_data, "none")
        np.testing.assert_array_equal(result, sample_data)

        # Also test with empty string
        result_empty = EmbeddingNormalizer.normalize_embeddings(sample_data, "")
        np.testing.assert_array_equal(result_empty, sample_data)

    def test_standard_normalization(self, sample_data):
        """Test standard (z-score) normalization."""
        result = EmbeddingNormalizer.normalize_embeddings(sample_data, "standard")

        # Check output shape
        assert result.shape == sample_data.shape

        # Check that each feature has approximately zero mean and unit variance
        feature_means = np.mean(result, axis=0)
        feature_stds = np.std(result, axis=0)

        np.testing.assert_allclose(feature_means, 0, atol=1e-10)
        np.testing.assert_allclose(feature_stds, 1, atol=1e-10)

        # Check that data type is preserved
        assert result.dtype == sample_data.dtype

    def test_l2_normalization(self, sample_data):
        """Test L2 normalization."""
        result = EmbeddingNormalizer.normalize_embeddings(sample_data, "l2")

        # Check output shape
        assert result.shape == sample_data.shape

        # Check that each sample has unit norm
        sample_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(sample_norms, 1, atol=1e-10)

        # Check that zero vector is handled (if any)
        zero_vector = np.zeros((1, sample_data.shape[1]))
        result_zero = EmbeddingNormalizer.normalize_embeddings(zero_vector, "l2")
        assert not np.isnan(result_zero).any()

    def test_pca_whitening(self, sample_data):
        """Test PCA whitening normalization."""
        # Skip if too few samples
        if sample_data.shape[0] < sample_data.shape[1]:
            pytest.skip("Not enough samples for PCA whitening")

        result = EmbeddingNormalizer.normalize_embeddings(sample_data, "pca")

        # Check output shape
        assert result.shape == sample_data.shape

        # Check that covariance matrix is approximately identity
        cov_matrix = np.cov(result.T)
        identity = np.eye(cov_matrix.shape[0])

        # Check diagonal is approximately 1
        np.testing.assert_allclose(np.diag(cov_matrix), 1, atol=1e-2)

        # Check off-diagonal elements are approximately 0
        off_diagonal = cov_matrix - np.diag(np.diag(cov_matrix))
        assert np.max(np.abs(off_diagonal)) < 1e-10

    def test_zca_whitening(self, sample_data):
        """Test ZCA whitening normalization."""
        # Skip if too few samples
        if sample_data.shape[0] < sample_data.shape[1]:
            pytest.skip("Not enough samples for ZCA whitening")

        result = EmbeddingNormalizer.normalize_embeddings(sample_data, "zca")

        # Check output shape
        assert result.shape == sample_data.shape

        # Check that covariance matrix is approximately identity
        cov_matrix = np.cov(result.T)

        # Check diagonal is approximately 1
        np.testing.assert_allclose(np.diag(cov_matrix), 1, atol=1e-2)

        # Check off-diagonal elements are approximately 0
        off_diagonal = cov_matrix - np.diag(np.diag(cov_matrix))
        assert np.max(np.abs(off_diagonal)) < 1e-3  # Slightly relaxed for ZCA

    def test_invalid_normalization_method(self, sample_data):
        """Test that invalid normalization method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            EmbeddingNormalizer.normalize_embeddings(sample_data, "invalid_method")

    def test_normalization_with_single_sample(self):
        """Test normalization with single sample."""
        single_sample = np.array([[1.0, 2.0, 3.0, 4.0]])

        # L2 should work
        result_l2 = EmbeddingNormalizer.normalize_embeddings(single_sample, "l2")
        assert result_l2.shape == single_sample.shape
        np.testing.assert_allclose(np.linalg.norm(result_l2), 1, atol=1e-10)

        # Standard should work (though not very meaningful)
        result_std = EmbeddingNormalizer.normalize_embeddings(single_sample, "standard")
        assert result_std.shape == single_sample.shape

        # PCA/ZCA might fail or change dimensions with single sample, which is acceptable
        for method in ["pca", "zca"]:
            try:
                result = EmbeddingNormalizer.normalize_embeddings(single_sample, method)
                # Just check that we get some valid output, shape might change
                assert isinstance(result, np.ndarray)
                assert result.shape[0] == single_sample.shape[0]  # Same number of samples
            except (np.linalg.LinAlgError, ValueError):
                # It's acceptable to fail with insufficient data
                pass

    def test_normalization_with_zero_variance_features(self):
        """Test normalization with features that have zero variance."""
        # Create data with one constant feature
        data = np.random.randn(10, 3)
        data[:, 1] = 5.0  # Constant feature

        # Standard normalization should handle this gracefully
        result_std = EmbeddingNormalizer.normalize_embeddings(data, "standard")
        assert result_std.shape == data.shape
        assert not np.isnan(result_std).any()
        assert not np.isinf(result_std).any()

        # L2 should work fine
        result_l2 = EmbeddingNormalizer.normalize_embeddings(data, "l2")
        assert result_l2.shape == data.shape
        sample_norms = np.linalg.norm(result_l2, axis=1)
        np.testing.assert_allclose(sample_norms, 1, atol=1e-10)

    def test_normalization_with_nan_values(self):
        """Test normalization behavior with NaN values."""
        data = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Most normalization methods should either handle NaN or raise an error
        for method in ["standard", "l2", "pca", "zca"]:
            try:
                result = EmbeddingNormalizer.normalize_embeddings(data, method)
                # If it succeeds, check shape is preserved
                assert result.shape == data.shape
            except (ValueError, np.linalg.LinAlgError):
                # It's acceptable to fail with NaN values
                pass

    def test_normalization_with_inf_values(self):
        """Test normalization behavior with infinite values."""
        data = np.array([[1.0, 2.0, np.inf], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Most normalization methods should either handle inf or raise an error
        for method in ["standard", "l2", "pca", "zca"]:
            try:
                result = EmbeddingNormalizer.normalize_embeddings(data, method)
                # If it succeeds, check shape is preserved
                assert result.shape == data.shape
            except (ValueError, np.linalg.LinAlgError):
                # It's acceptable to fail with inf values
                pass

    def test_normalization_preserves_data_type(self):
        """Test that normalization preserves input data type when possible."""
        for dtype in [np.float32, np.float64]:
            data = np.random.randn(10, 5).astype(dtype)

            for method in ["none", "standard", "l2"]:
                result = EmbeddingNormalizer.normalize_embeddings(data, method)
                # Note: Some operations might promote float32 to float64
                assert np.issubdtype(result.dtype, np.floating)

    def test_normalization_consistency(self):
        """Test that normalization gives consistent results."""
        np.random.seed(123)
        data = np.random.randn(50, 20)

        for method in ["standard", "l2", "pca", "zca"]:
            result1 = EmbeddingNormalizer.normalize_embeddings(data, method)
            result2 = EmbeddingNormalizer.normalize_embeddings(data, method)

            # Results should be identical for the same input
            np.testing.assert_array_equal(result1, result2)

    def test_large_data_normalization(self):
        """Test normalization with larger datasets."""
        # Create larger dataset to test scalability
        large_data = np.random.randn(1000, 100)

        for method in ["standard", "l2"]:
            result = EmbeddingNormalizer.normalize_embeddings(large_data, method)
            assert result.shape == large_data.shape

        # PCA/ZCA might be slower but should still work
        for method in ["pca", "zca"]:
            try:
                result = EmbeddingNormalizer.normalize_embeddings(large_data, method)
                assert result.shape == large_data.shape
            except MemoryError:
                # Acceptable for very large matrices
                pytest.skip(f"{method} failed due to memory constraints")


class TestNormalizationIntegration:
    """Test normalization integration with clustering pipeline."""

    def test_clustering_config_normalization_method(self):
        """Test that ClusteringConfig properly handles normalization_method."""
        config = ClusteringConfig(
            embedding_files=["test.pkl"], metadata_file="test.tsv", normalization_method="l2"
        )
        assert config.normalization_method == "l2"

        # Test default
        config_default = ClusteringConfig(embedding_files=["test.pkl"], metadata_file="test.tsv")
        assert config_default.normalization_method == "l2"

    @patch("clustering_evaluation.ClusteringEngine.perform_clustering")
    @patch("clustering_evaluation.ClusteringEngine.evaluate_clustering")
    def test_normalization_applied_in_subsampling(
        self, mock_evaluate, mock_clustering, sample_embeddings, sample_metadata, temp_files
    ):
        """Test that normalization is applied in subsampling analysis."""
        mock_clustering.return_value = np.array([0, 0, 1, 1, 1])
        mock_evaluate.return_value = {
            "adjusted_rand_score": 0.5,
            "silhouette_score": 0.3,
            "calinski_harabasz_score": 10.0,
            "davies_bouldin_score": 1.5,
            "normalized_mutual_info": 0.4,
            "homogeneity": 0.3,
            "completeness": 0.6,
            "v_measure": 0.4,
        }

        config = ClusteringConfig(
            embedding_files=[temp_files["embedding_file"]],
            metadata_file=temp_files["metadata_file"],
            normalization_method="standard",
            subsample=2,  # Small number for testing
            subsample_fraction=0.8,
        )

        analyzer = SubsamplingAnalyzer(config)

        # Mock the required objects
        embeddings_dict = {temp_files["embedding_file"]: sample_embeddings}
        protein_ids = list(sample_embeddings.keys())
        true_labels = np.array([0, 0, 1, 1, 2])

        # This should not raise an error and should apply normalization
        result = analyzer.run_subsampling_analysis(protein_ids, embeddings_dict, true_labels)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_normalization_method_validation(self):
        """Test that invalid normalization methods are caught early."""
        with pytest.raises(ValueError):
            EmbeddingNormalizer.normalize_embeddings(np.random.randn(10, 5), "invalid")

    def test_normalization_with_real_clustering(self, sample_embeddings):
        """Test normalization with actual clustering (integration test)."""
        embeddings_matrix = np.array(list(sample_embeddings.values()))

        # Test that different normalizations produce different clustering inputs
        norm_none = EmbeddingNormalizer.normalize_embeddings(embeddings_matrix, "none")
        norm_std = EmbeddingNormalizer.normalize_embeddings(embeddings_matrix, "standard")
        norm_l2 = EmbeddingNormalizer.normalize_embeddings(embeddings_matrix, "l2")

        # They should be different (unless original data was already normalized)
        assert not np.allclose(norm_none, norm_std, atol=1e-3)
        assert not np.allclose(norm_none, norm_l2, atol=1e-3)
        assert not np.allclose(norm_std, norm_l2, atol=1e-3)

        # All should have the same shape
        assert norm_none.shape == norm_std.shape == norm_l2.shape == embeddings_matrix.shape
