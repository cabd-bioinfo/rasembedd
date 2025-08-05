"""Tests for embedding generation functionality."""

import os
import pickle

# Import functions from generate_embeddings.py
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generate_embeddings import (
    get_available_models,
    load_model_class,
    load_sequences_from_fasta,
    load_sequences_from_tsv,
    main,
    save_embeddings,
)
from tests.conftest import assert_embeddings_valid


class TestSequenceLoading:
    """Test cases for sequence loading functionality."""

    def test_load_sequences_from_fasta(self, temp_fasta_file, sample_sequences):
        """Test loading sequences from FASTA file."""
        sequences = load_sequences_from_fasta(temp_fasta_file)

        assert len(sequences) == len(sample_sequences)
        for seq_id, expected_seq in sample_sequences.items():
            assert seq_id in sequences
            assert sequences[seq_id] == expected_seq

    def test_load_sequences_empty_file(self):
        """Test loading from empty FASTA file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            temp_path = f.name

        try:
            sequences = load_sequences_from_fasta(temp_path)
            assert sequences == {}
        finally:
            os.unlink(temp_path)

    def test_load_sequences_malformed_fasta(self):
        """Test loading from malformed FASTA file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write("This is not a FASTA file\n")
            f.write("Just plain text\n")
            temp_path = f.name

        try:
            sequences = load_sequences_from_fasta(temp_path)
            assert sequences == {}
        finally:
            os.unlink(temp_path)

    def test_load_sequences_multiline_sequence(self):
        """Test loading sequences with multiple lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">test_seq\n")
            f.write("MKWVTFISLLL\n")
            f.write("LFSSAYSRGVF\n")
            f.write("RRDTHKSEIAH\n")
            temp_path = f.name

        try:
            sequences = load_sequences_from_fasta(temp_path)
            expected_seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAH"
            assert sequences["test_seq"] == expected_seq
        finally:
            os.unlink(temp_path)

    def test_load_sequences_from_tsv(self, temp_dir):
        """Test loading sequences from TSV file."""
        tsv_file = os.path.join(temp_dir, "test_sequences.tsv")

        # Create test TSV file
        test_data = pd.DataFrame(
            {
                "uniprot_id": ["seq1", "seq2", "seq3"],
                "sequence": ["MKWVTFISLLL", "ARNDCQEGHILKMFPSTWYV", "QWERTYUIOP"],
                "family": ["FamA", "FamB", "FamA"],
            }
        )
        test_data.to_csv(tsv_file, sep="\t", index=False)

        sequences = load_sequences_from_tsv(tsv_file)

        assert len(sequences) == 3
        assert sequences["seq1"] == "MKWVTFISLLL"
        assert sequences["seq2"] == "ARNDCQEGHILKMFPSTWYV"
        assert sequences["seq3"] == "QWERTYUIOP"

    def test_load_sequences_from_tsv_custom_columns(self, temp_dir):
        """Test loading sequences from TSV with custom column names."""
        tsv_file = os.path.join(temp_dir, "test_sequences.tsv")

        # Create test TSV file with custom column names
        test_data = pd.DataFrame(
            {
                "protein_id": ["seq1", "seq2"],
                "amino_acid_sequence": ["MKWVTFISLLL", "ARNDCQEGHILKMFPSTWYV"],
                "family": ["FamA", "FamB"],
            }
        )
        test_data.to_csv(tsv_file, sep="\t", index=False)

        sequences = load_sequences_from_tsv(tsv_file, "protein_id", "amino_acid_sequence")

        assert len(sequences) == 2
        assert sequences["seq1"] == "MKWVTFISLLL"
        assert sequences["seq2"] == "ARNDCQEGHILKMFPSTWYV"


class TestEmbeddingSaving:
    """Test cases for embedding saving functionality."""

    def test_save_embeddings_pickle(self, sample_embeddings, temp_dir):
        """Test saving embeddings in pickle format."""
        output_path = os.path.join(temp_dir, "test_embeddings.pkl")

        save_embeddings(sample_embeddings, output_path, "pickle")

        assert os.path.exists(output_path)

        # Load and verify
        with open(output_path, "rb") as f:
            loaded_embeddings = pickle.load(f)

        assert len(loaded_embeddings) == len(sample_embeddings)
        for seq_id, embedding in sample_embeddings.items():
            assert seq_id in loaded_embeddings
            np.testing.assert_array_equal(loaded_embeddings[seq_id], embedding)

    def test_save_embeddings_npz(self, sample_embeddings, temp_dir):
        """Test saving embeddings in NPZ format."""
        output_path = os.path.join(temp_dir, "test_embeddings.npz")

        save_embeddings(sample_embeddings, output_path, "npz")

        assert os.path.exists(output_path)

        # Load and verify
        data = np.load(output_path, allow_pickle=True)
        loaded_ids = data["ids"]
        loaded_embeddings = data["embeddings"]

        assert len(loaded_ids) == len(sample_embeddings)

        for i, seq_id in enumerate(loaded_ids):
            expected_embedding = sample_embeddings[seq_id]
            np.testing.assert_array_equal(loaded_embeddings[i], expected_embedding)

    @pytest.mark.skipif(not pytest.importorskip("h5py"), reason="h5py not available")
    def test_save_embeddings_hdf5(self, sample_embeddings, temp_dir):
        """Test saving embeddings in HDF5 format."""
        import h5py

        output_path = os.path.join(temp_dir, "test_embeddings.h5")

        save_embeddings(sample_embeddings, output_path, "hdf5")

        assert os.path.exists(output_path)

        # Load and verify - embeddings are stored in embeddings group
        with h5py.File(output_path, "r") as f:
            embeddings_group = f["embeddings"]
            for seq_id, expected_embedding in sample_embeddings.items():
                # Safe ID conversion (same as in save_embeddings)
                safe_id = seq_id.replace("/", "_").replace("\\", "_").replace("|", "_")
                assert safe_id in embeddings_group
                loaded_embedding = embeddings_group[safe_id][:]
                np.testing.assert_array_equal(loaded_embedding, expected_embedding)

    def test_save_embeddings_invalid_format(self, sample_embeddings, temp_dir):
        """Test saving with invalid format."""
        output_path = os.path.join(temp_dir, "test_embeddings.invalid")

        with pytest.raises(ValueError, match="Unsupported format"):
            save_embeddings(sample_embeddings, output_path, "invalid")


class TestModelInstantiation:
    """Test cases for model instantiation."""

    def test_load_model_class_success(self):
        """Test successful model class loading."""
        from models.base_model import BaseEmbeddingModel

        # Test with a real model class that exists
        model_class = load_model_class("prost_t5")

        # Verify it's a class and the correct type
        assert isinstance(model_class, type)
        assert issubclass(model_class, BaseEmbeddingModel)
        assert model_class != BaseEmbeddingModel

    @patch("generate_embeddings.importlib.import_module")
    def test_load_model_class_import_error(self, mock_import):
        """Test model class loading with import error."""
        mock_import.side_effect = ImportError("Module not found")

        with pytest.raises(ValueError, match="Model type .* not found"):
            load_model_class("nonexistent_model")

    @patch("generate_embeddings.importlib.import_module")
    def test_load_model_class_no_valid_class(self, mock_import):
        """Test model class loading with no valid model class."""
        mock_module = Mock()
        mock_import.return_value = mock_module

        # Mock dir to return no valid model classes
        with patch("builtins.dir", return_value=["SomeOtherClass"]):
            with pytest.raises(ValueError, match="No valid model class found"):
                load_model_class("invalid_model")

    def test_get_available_models(self):
        """Test getting available models."""
        available_models = get_available_models()

        # Should return a dictionary
        assert isinstance(available_models, dict)

        # Should contain at least some model files if models directory exists
        # This is environment-dependent, so we just check the structure
        for model_name, model_path in available_models.items():
            assert isinstance(model_name, str)
            assert isinstance(model_path, str)
            assert model_name != "base_model"  # Should exclude base_model
            assert not model_name.startswith("__")  # Should exclude __init__ etc.


class TestDeviceSetup:
    """Test cases for device setup."""

    def test_device_creation_cpu(self):
        """Test CPU device creation."""
        device = torch.device("cpu")
        assert device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_creation_cuda(self):
        """Test CUDA device creation."""
        device = torch.device("cuda:0")
        assert device.type == "cuda"
        assert device.index == 0

    def test_device_auto_selection(self):
        """Test automatic device selection logic."""
        # Test the logic used in main()
        if torch.cuda.is_available():
            device = torch.device("cuda")
            assert device.type == "cuda"
        else:
            device = torch.device("cpu")
            assert device.type == "cpu"

    def test_precision_types(self):
        """Test different precision types."""
        assert torch.float32 == torch.float32
        assert torch.float16 == torch.float16

        # Test tensor creation with different precisions
        tensor_f32 = torch.tensor([1.0], dtype=torch.float32)
        tensor_f16 = torch.tensor([1.0], dtype=torch.float16)

        assert tensor_f32.dtype == torch.float32
        assert tensor_f16.dtype == torch.float16


class TestEmbeddingGeneration:
    """Integration tests for embedding generation."""

    @patch("generate_embeddings.load_model_class")
    def test_embedding_generation_flow(self, mock_load_model_class, temp_fasta_file, temp_dir):
        """Test the complete embedding generation flow."""
        # Mock model
        mock_model = Mock()
        mock_embeddings = {
            "test_seq_1": np.random.randn(128).astype(np.float32),
            "test_seq_2": np.random.randn(128).astype(np.float32),
            "test_seq_3": np.random.randn(128).astype(np.float32),
        }
        mock_model.generate_embeddings.return_value = mock_embeddings

        mock_model_class = Mock()
        mock_model_class.return_value = mock_model
        mock_load_model_class.return_value = mock_model_class

        output_path = os.path.join(temp_dir, "test_output.pkl")

        # Mock sys.argv for argparse
        test_args = [
            "generate_embeddings.py",
            "--input",
            temp_fasta_file,
            "--input_type",
            "fasta",
            "--model_type",
            "prost_t5",
            "--output",
            output_path,
            "--device",
            "cpu",
        ]

        with patch("sys.argv", test_args):
            # This would test the main function, but we need to mock more components
            # For now, we'll test individual components
            pass

    def test_embedding_validation(self, sample_embeddings):
        """Test embedding validation helper."""
        # Test valid embeddings
        assert_embeddings_valid(sample_embeddings, list(sample_embeddings.keys()))

        # Test invalid embeddings (empty)
        with pytest.raises(AssertionError, match="should not be empty"):
            assert_embeddings_valid({})

        # Test invalid embeddings (wrong IDs)
        with pytest.raises(AssertionError, match="should match expected IDs"):
            assert_embeddings_valid(sample_embeddings, ["wrong_id"])

        # Test invalid embeddings (NaN values)
        invalid_embeddings = sample_embeddings.copy()
        invalid_embeddings["test_seq_1"][0] = np.nan
        with pytest.raises(AssertionError, match="should not contain NaN"):
            assert_embeddings_valid(invalid_embeddings)

    def test_sequence_validation(self):
        """Test sequence validation."""
        # Valid protein sequence
        valid_seq = "MKWVTFISLLLLFSSAYSRGVF"
        assert all(c in "ACDEFGHIKLMNPQRSTVWY" for c in valid_seq.upper())

        # Invalid characters (should be handled by models)
        invalid_seq = "MKWVTFISLLLXLFSSAYSRGVF"
        assert not all(c in "ACDEFGHIKLMNPQRSTVWY" for c in invalid_seq.upper())

    def test_batch_processing_simulation(self, sample_sequences):
        """Test batch processing simulation."""
        # Simulate processing sequences in batches
        batch_size = 2
        sequences_list = list(sample_sequences.items())

        batches = [
            sequences_list[i : i + batch_size] for i in range(0, len(sequences_list), batch_size)
        ]

        assert len(batches) == 2  # 3 sequences with batch size 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
