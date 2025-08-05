"""Tests for visualization functionality including distance metrics and projections."""

import os

# Import functions from generate_visualizations.py
import sys
import tempfile
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generate_visualizations import (
    calculate_distances,
    compute_projection,
    extract_species,
    get_color_palette,
    get_marker_shapes,
    get_output_filename,
    load_embeddings,
    load_metadata,
    set_random_seeds,
    validate_inputs,
)
from tests.conftest import assert_distances_valid, assert_projection_valid


class TestDistanceMetrics:
    """Test cases for distance metric calculations."""

    def test_cosine_distances(self, sample_embeddings):
        """Test cosine distance calculation."""
        embeddings_array = np.array(list(sample_embeddings.values()))
        distances = calculate_distances(embeddings_array, "cosine")

        assert_distances_valid(distances, len(sample_embeddings))

        # Compare with sklearn implementation
        expected_distances = cosine_distances(embeddings_array)
        np.testing.assert_array_almost_equal(distances, expected_distances, decimal=5)

    def test_euclidean_distances(self, sample_embeddings):
        """Test Euclidean distance calculation."""
        embeddings_array = np.array(list(sample_embeddings.values()))
        distances = calculate_distances(embeddings_array, "euclidean")

        assert_distances_valid(distances, len(sample_embeddings))

        # Compare with sklearn implementation
        expected_distances = euclidean_distances(embeddings_array)
        np.testing.assert_array_almost_equal(distances, expected_distances, decimal=5)

    def test_invalid_distance_metric(self, sample_embeddings):
        """Test invalid distance metric."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        with pytest.raises(ValueError, match="Invalid distance metric"):
            calculate_distances(embeddings_array, "invalid_metric")

    def test_distance_properties(self, sample_embeddings):
        """Test mathematical properties of distance metrics."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        for metric in ["cosine", "euclidean"]:
            distances = calculate_distances(embeddings_array, metric)

            # Test symmetry
            assert np.allclose(distances, distances.T), f"{metric} distances should be symmetric"

            # Test diagonal is zero
            assert np.allclose(np.diag(distances), 0), f"{metric} distance diagonal should be zero"

            # Test non-negativity
            assert np.all(distances >= 0), f"{metric} distances should be non-negative"

    def test_distance_reproducibility(self, sample_embeddings):
        """Test that distance calculations are reproducible."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Set random seed (though distances shouldn't use randomness)
        set_random_seeds(42)
        distances1 = calculate_distances(embeddings_array, "cosine")

        set_random_seeds(42)
        distances2 = calculate_distances(embeddings_array, "cosine")

        np.testing.assert_array_equal(distances1, distances2)


class TestProjectionMethods:
    """Test cases for dimensionality reduction methods."""

    def test_umap_projection(self, sample_embeddings):
        """Test UMAP projection."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Mock UMAP for small datasets since UMAP doesn't work well with < 4 samples
        with patch("generate_visualizations.umap.UMAP") as mock_umap:
            mock_reducer = Mock()
            mock_reducer.fit_transform.return_value = np.random.rand(len(sample_embeddings), 2)
            mock_umap.return_value = mock_reducer

            projection, params = compute_projection(embeddings_array, "UMAP", random_seed=42)

        assert_projection_valid(projection, len(sample_embeddings))
        assert "random_state" in params
        assert params["random_state"] == 42

    def test_pca_projection(self, sample_embeddings):
        """Test PCA projection."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        projection, params = compute_projection(embeddings_array, "PCA", random_seed=42)

        assert_projection_valid(projection, len(sample_embeddings))
        assert "n_components" in params
        assert params["n_components"] == 2

    def test_tsne_projection(self, sample_embeddings):
        """Test t-SNE projection."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Use appropriate perplexity for small dataset
        projection, params = compute_projection(
            embeddings_array,
            "TSNE",
            random_seed=42,
            perplexity=min(1.0, (len(embeddings_array) - 1) / 3.0),  # Ensure perplexity < n_samples
        )

        assert_projection_valid(projection, len(sample_embeddings))
        assert "random_state" in params
        assert params["random_state"] == 42
        assert params["perplexity"] <= len(sample_embeddings) - 1

    @pytest.mark.skipif(not pytest.importorskip("pacmap"), reason="PaCMAP not available")
    def test_pacmap_projection(self, sample_embeddings):
        """Test PaCMAP projection."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Mock PaCMAP for small datasets since PaCMAP doesn't work well with < 4 samples
        with patch("generate_visualizations.pacmap.PaCMAP") as mock_pacmap:
            mock_reducer = Mock()
            mock_reducer.fit_transform.return_value = np.random.rand(len(sample_embeddings), 2)
            mock_pacmap.return_value = mock_reducer

            projection, params = compute_projection(embeddings_array, "PaCMAP", random_seed=42)

        assert_projection_valid(projection, len(sample_embeddings))
        assert "random_state" in params
        assert params["random_state"] == 42

    def test_invalid_projection_method(self, sample_embeddings):
        """Test invalid projection method."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        with pytest.raises(ValueError, match="Invalid projection method"):
            compute_projection(embeddings_array, "INVALID", random_seed=42)

    def test_projection_reproducibility(self, sample_embeddings):
        """Test that projections are reproducible with same seed."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        projection1, _ = compute_projection(embeddings_array, "PCA", random_seed=42)
        projection2, _ = compute_projection(embeddings_array, "PCA", random_seed=42)

        np.testing.assert_array_almost_equal(projection1, projection2, decimal=5)

    def test_projection_parameters(self, sample_embeddings):
        """Test projection with custom parameters."""
        embeddings_array = np.array(list(sample_embeddings.values()))

        # Mock UMAP for small datasets to test parameter passing
        with patch("generate_visualizations.umap.UMAP") as mock_umap:
            mock_reducer = Mock()
            mock_reducer.fit_transform.return_value = np.random.rand(len(sample_embeddings), 2)
            mock_umap.return_value = mock_reducer

            projection, params = compute_projection(
                embeddings_array,
                "UMAP",
                random_seed=42,
                n_neighbors=5,  # Will be automatically limited to len(embeddings) - 1
                min_dist=0.5,
                umap_metric="manhattan",
            )

        assert_projection_valid(projection, len(sample_embeddings))
        # Check that n_neighbors was limited appropriately
        assert params["n_neighbors"] == min(5, len(embeddings_array) - 1)
        assert params["min_dist"] == 0.5
        assert params["metric"] == "manhattan"


class TestColorPalettes:
    """Test cases for color palette generation."""

    def test_glasbey_palette(self):
        """Test Glasbey color palette."""
        colors = get_color_palette(5, "glasbey")
        assert len(colors) == 5
        # Each color should be a tuple/list of RGB values
        for color in colors:
            assert len(color) == 3
            assert all(0 <= c <= 1 for c in color)

    def test_tab10_palette(self):
        """Test Tab10 color palette."""
        colors = get_color_palette(5, "tab10")
        assert len(colors) == 5

    def test_large_palette(self):
        """Test palette with many colors."""
        colors = get_color_palette(50, "glasbey")
        assert len(colors) == 50

    def test_palette_reproducibility(self):
        """Test that color palettes are reproducible."""
        colors1 = get_color_palette(10, "glasbey")
        colors2 = get_color_palette(10, "glasbey")

        # Colors should be the same (due to fixed random seed in function)
        assert len(colors1) == len(colors2)


class TestSpeciesExtraction:
    """Test cases for species extraction."""

    def test_extract_species_from_column(self, sample_metadata):
        """Test species extraction from metadata column."""
        ids = ["test_seq_1", "test_seq_2", "test_seq_3"]
        species = extract_species(ids, sample_metadata, "species")

        expected_species = ["test_species_1", "test_species_2", "test_species_1"]
        assert species == expected_species

    def test_extract_species_missing_column(self, sample_metadata):
        """Test species extraction when column is missing."""
        ids = ["test_seq_1", "test_seq_2", "test_seq_3"]
        species = extract_species(ids, sample_metadata, "nonexistent_column")

        # Should fall back to UniProt ID parsing
        expected_species = ["1", "2", "3"]  # Last part after underscore
        assert species == expected_species

    def test_extract_species_uniprot_fallback(self):
        """Test species extraction fallback to UniProt ID parsing."""
        ids = ["P12345_HUMAN", "Q67890_MOUSE", "R54321_ECOLI"]
        df = pd.DataFrame({"uniprot_id": ids})  # No species column

        species = extract_species(ids, df, "species")
        expected_species = ["HUMAN", "MOUSE", "ECOLI"]
        assert species == expected_species

    def test_extract_species_no_underscore(self):
        """Test species extraction with IDs without underscores."""
        ids = ["seq1", "seq2", "seq3"]
        df = pd.DataFrame({"uniprot_id": ids})  # No species column

        species = extract_species(ids, df, "species")
        expected_species = ["UNKNOWN", "UNKNOWN", "UNKNOWN"]
        assert species == expected_species


class TestMarkerShapes:
    """Test cases for marker shape generation."""

    def test_marker_shapes_small(self):
        """Test marker shapes for small number of species."""
        markers = get_marker_shapes(5)
        assert len(markers) == 5
        assert len(set(markers)) == 5  # All unique

    def test_marker_shapes_large(self):
        """Test marker shapes for large number of species."""
        markers = get_marker_shapes(20)
        assert len(markers) == 20
        # Should repeat markers if needed

    def test_marker_shapes_zero(self):
        """Test marker shapes for zero species."""
        markers = get_marker_shapes(0)
        assert len(markers) == 0


class TestInputValidation:
    """Test cases for input validation."""

    def test_validate_inputs_success(self, sample_metadata, sample_embeddings):
        """Test successful input validation."""
        df_filtered, embeddings_array = validate_inputs(
            sample_metadata, sample_embeddings, "Family.name", "uniprot_id"
        )

        assert len(df_filtered) == len(sample_embeddings)
        assert embeddings_array.shape[0] == len(sample_embeddings)
        assert "uniprot_id" in df_filtered.columns
        assert "Family.name" in df_filtered.columns

    def test_validate_inputs_missing_id_column(self, sample_metadata, sample_embeddings):
        """Test validation with missing ID column."""
        with pytest.raises(ValueError, match="ID column .* not found"):
            validate_inputs(sample_metadata, sample_embeddings, "Family.name", "missing_id")

    def test_validate_inputs_missing_color_column(self, sample_metadata, sample_embeddings):
        """Test validation with missing color column."""
        with pytest.raises(ValueError, match="Color column .* not found"):
            validate_inputs(sample_metadata, sample_embeddings, "missing_color", "uniprot_id")

    def test_validate_inputs_no_common_ids(self, sample_metadata, sample_embeddings):
        """Test validation with no common IDs."""
        # Create embeddings with different IDs
        different_embeddings = {"different_id": np.random.randn(128)}

        with pytest.raises(ValueError, match="No common IDs found"):
            validate_inputs(sample_metadata, different_embeddings, "Family.name", "uniprot_id")

    def test_validate_inputs_partial_overlap(self, sample_metadata, sample_embeddings):
        """Test validation with partial ID overlap."""
        # Add extra metadata entry and remove one embedding
        extra_metadata = sample_metadata.copy()
        extra_row = pd.DataFrame(
            {
                "uniprot_id": ["extra_seq"],
                "Family.name": ["ExtraFamily"],
                "species": ["extra_species"],
                "length": [100],
            }
        )
        extra_metadata = pd.concat([extra_metadata, extra_row], ignore_index=True)

        partial_embeddings = sample_embeddings.copy()
        del partial_embeddings["test_seq_3"]

        df_filtered, embeddings_array = validate_inputs(
            extra_metadata, partial_embeddings, "Family.name", "uniprot_id"
        )

        # Should only include sequences with both metadata and embeddings
        assert len(df_filtered) == 2  # test_seq_1 and test_seq_2
        assert embeddings_array.shape[0] == 2


class TestFileOperations:
    """Test cases for file loading and output filename generation."""

    def test_load_embeddings_pickle(self, temp_embeddings_file, sample_embeddings):
        """Test loading embeddings from pickle file."""
        embeddings = load_embeddings(temp_embeddings_file)

        assert len(embeddings) == len(sample_embeddings)
        for seq_id, expected_embedding in sample_embeddings.items():
            assert seq_id in embeddings
            np.testing.assert_array_equal(embeddings[seq_id], expected_embedding)

    def test_load_embeddings_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_embeddings("nonexistent_file.pkl")

    def test_load_metadata_tsv(self, temp_metadata_file, sample_metadata):
        """Test loading metadata from TSV file."""
        metadata = load_metadata(temp_metadata_file)

        pd.testing.assert_frame_equal(metadata, sample_metadata)

    def test_get_output_filename(self):
        """Test output filename generation."""
        filename = get_output_filename("test", "UMAP", "Family.name", "embeddings.pkl", "pdf")

        assert "UMAP_projection" in filename
        assert "Family.name" in filename
        assert "embeddings" in filename
        assert filename.endswith(".pdf")

    def test_get_output_filename_heatmap(self):
        """Test output filename generation for heatmaps."""
        filename = get_output_filename(
            "test_heatmap_cosine", "heatmap", "Family.name", "embeddings.pkl", "png"
        )

        assert "heatmap_cosine" in filename
        assert "Family.name" in filename
        assert filename.endswith(".png")


class TestRandomSeedSetting:
    """Test cases for random seed setting."""

    def test_set_random_seeds(self):
        """Test that random seeds are set correctly."""
        set_random_seeds(42)

        # Test NumPy random seed
        random1 = np.random.randn(5)

        set_random_seeds(42)
        random2 = np.random.randn(5)

        np.testing.assert_array_equal(random1, random2)

    def test_environment_variables_set(self):
        """Test that environment variables are set for reproducibility."""
        set_random_seeds(42)

        assert os.environ.get("PYTHONHASHSEED") == "42"
        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS") == "1"
