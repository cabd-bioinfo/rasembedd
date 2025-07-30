"""Shared test fixtures and utilities."""

import os
import pickle
import tempfile
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest

# Test data constants
SAMPLE_SEQUENCES = {
    "test_seq_1": "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLQVLAQLTE",
    "test_seq_2": "MRLTVLALLLLAAGLLQGPGGEEGAGAGGLDGLVLSGPFPLAPLAGGWWLAAAAAGGGGGSGGGGAGGGGGGGGSSSWSRGGGGGRGRGGGRGGGRGRGSSGRSSSGGGGGSGGGRGGGGGGGGGGGGGGGGGGS",
    "test_seq_3": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
}

SAMPLE_METADATA = pd.DataFrame(
    {
        "uniprot_id": ["test_seq_1", "test_seq_2", "test_seq_3"],
        "Family.name": ["TestFamily1", "TestFamily2", "TestFamily1"],
        "species": ["test_species_1", "test_species_2", "test_species_1"],
        "length": [120, 130, 110],
    }
)


@pytest.fixture
def sample_sequences():
    """Provide sample protein sequences for testing."""
    return SAMPLE_SEQUENCES.copy()


@pytest.fixture
def sample_metadata():
    """Provide sample metadata for testing."""
    return SAMPLE_METADATA.copy()


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings (random but reproducible)."""
    np.random.seed(42)  # For reproducible tests
    embeddings = {}
    for seq_id in SAMPLE_SEQUENCES.keys():
        # Create random embeddings with realistic dimensions
        embeddings[seq_id] = np.random.randn(1024).astype(np.float32)
    return embeddings


@pytest.fixture
def temp_fasta_file(sample_sequences):
    """Create a temporary FASTA file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        for seq_id, sequence in sample_sequences.items():
            f.write(f">{seq_id}\n{sequence}\n")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_metadata_file(sample_metadata):
    """Create a temporary metadata file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        sample_metadata.to_csv(f.name, sep="\t", index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_embeddings_file(sample_embeddings):
    """Create a temporary embeddings pickle file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        with open(f.name, "wb") as pkl_f:
            pickle.dump(sample_embeddings, pkl_f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path

    # Cleanup
    import shutil

    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


def assert_embeddings_valid(embeddings: Dict[str, np.ndarray], expected_ids: list = None):
    """Assert that embeddings dictionary is valid."""
    assert isinstance(embeddings, dict), "Embeddings should be a dictionary"
    assert len(embeddings) > 0, "Embeddings should not be empty"

    if expected_ids:
        assert set(embeddings.keys()) == set(
            expected_ids
        ), "Embedding IDs should match expected IDs"

    for seq_id, embedding in embeddings.items():
        assert isinstance(embedding, np.ndarray), f"Embedding for {seq_id} should be numpy array"
        assert embedding.ndim == 1, f"Embedding for {seq_id} should be 1D array"
        assert embedding.dtype in [
            np.float32,
            np.float64,
        ], f"Embedding for {seq_id} should be float type"
        assert not np.isnan(
            embedding
        ).any(), f"Embedding for {seq_id} should not contain NaN values"
        assert not np.isinf(
            embedding
        ).any(), f"Embedding for {seq_id} should not contain inf values"


def assert_distances_valid(distances: np.ndarray, n_sequences: int):
    """Assert that distance matrix is valid."""
    assert isinstance(distances, np.ndarray), "Distances should be numpy array"
    assert distances.shape == (
        n_sequences,
        n_sequences,
    ), f"Distance matrix should be {n_sequences}x{n_sequences}"
    assert distances.dtype in [
        np.float32,
        np.float64,
    ], "Distance matrix should be float type"
    assert not np.isnan(distances).any(), "Distance matrix should not contain NaN values"
    assert not np.isinf(distances).any(), "Distance matrix should not contain inf values"

    # Check symmetry
    np.testing.assert_array_almost_equal(
        distances, distances.T, decimal=5, err_msg="Distance matrix should be symmetric"
    )

    # Check diagonal is zero (for most distance metrics)
    np.testing.assert_array_almost_equal(
        np.diag(distances),
        np.zeros(n_sequences),
        decimal=5,
        err_msg="Distance matrix diagonal should be zero",
    )


def assert_projection_valid(projection: np.ndarray, n_sequences: int, n_components: int = 2):
    """Assert that projection is valid."""
    assert isinstance(projection, np.ndarray), "Projection should be numpy array"
    assert projection.shape == (
        n_sequences,
        n_components,
    ), f"Projection should be {n_sequences}x{n_components}"
    assert projection.dtype in [
        np.float32,
        np.float64,
    ], "Projection should be float type"
    assert not np.isnan(projection).any(), "Projection should not contain NaN values"
    assert not np.isinf(projection).any(), "Projection should not contain inf values"
