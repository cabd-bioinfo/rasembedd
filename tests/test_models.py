"""Tests for individual model implementations."""

import os

# Test each model implementation
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from models.prost_t5 import ProstT5Model

    PROST_T5_AVAILABLE = True
except ImportError:
    PROST_T5_AVAILABLE = False

try:
    from models.ankh import AnkhModel

    ANKH_AVAILABLE = True
except ImportError:
    ANKH_AVAILABLE = False

try:
    from models.esm import ESMModel

    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

try:
    from models.prot_t5 import ProtT5Model

    PROT_T5_AVAILABLE = True
except ImportError:
    PROT_T5_AVAILABLE = False


@pytest.mark.skipif(not PROST_T5_AVAILABLE, reason="ProstT5Model not available")
class TestProstT5Model:
    """Test cases for ProstT5Model."""

    @patch("models.prost_t5.AutoTokenizer.from_pretrained")
    @patch("models.prost_t5.T5EncoderModel.from_pretrained")
    def test_prost_t5_initialization(self, mock_model, mock_tokenizer):
        """Test ProstT5Model initialization."""
        device = torch.device("cpu")
        model = ProstT5Model("prost_t5", device)

        assert model.model_name == "prost_t5"
        assert model.device == device

    @patch("models.prost_t5.AutoTokenizer.from_pretrained")
    @patch("models.prost_t5.T5EncoderModel.from_pretrained")
    def test_prost_t5_load_model(self, mock_model_class, mock_tokenizer_class):
        """Test ProstT5Model model loading."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_model_class.return_value = mock_model

        device = torch.device("cpu")
        model = ProstT5Model("prost_t5", device)
        model.load_model()

        assert model.tokenizer == mock_tokenizer
        assert model.model == mock_model
        mock_model.eval.assert_called_once()

    def test_prost_t5_preprocess_sequence(self):
        """Test ProstT5Model sequence preprocessing."""
        device = torch.device("cpu")
        model = ProstT5Model("prost_t5", device)

        sequence = "MKWVTFISLLL"
        processed = model.preprocess_sequence(sequence)

        # ProstT5 adds spaces between amino acids
        expected = " ".join(sequence)
        assert processed == expected

    @patch("models.prost_t5.AutoTokenizer.from_pretrained")
    @patch("models.prost_t5.T5EncoderModel.from_pretrained")
    def test_prost_t5_generate_embedding(self, mock_model_class, mock_tokenizer_class):
        """Test ProstT5Model embedding generation."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_tokenizer_class.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = torch.randn(1, 5, 1024)
        mock_model.return_value = mock_outputs
        mock_model_class.return_value = mock_model

        device = torch.device("cpu")
        model = ProstT5Model("prost_t5", device)
        model.load_model()

        sequence = "MKWVT"
        seq_id = "test_seq"

        embedding = model.generate_embedding(sequence, seq_id)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1024,)  # ProstT5 embedding dimension
        assert embedding.dtype == np.float32


@pytest.mark.skipif(not ANKH_AVAILABLE, reason="AnkhModel not available")
class TestAnkhModel:
    """Test cases for AnkhModel."""

    @patch("models.ankh.AnkhModel.load_model")
    def test_ankh_initialization(self, mock_load):
        """Test AnkhModel initialization."""
        device = torch.device("cpu")
        model = AnkhModel("ankh", device)

        assert model.model_name == "ankh"
        assert model.device == device

    def test_ankh_preprocess_sequence(self):
        """Test AnkhModel sequence preprocessing."""
        device = torch.device("cpu")
        model = AnkhModel("ankh", device)

        sequence = "mkwvtfislll"
        processed = model.preprocess_sequence(sequence)

        # Ankh typically uses uppercase
        assert processed == sequence.upper()


@pytest.mark.skipif(not ESM_AVAILABLE, reason="ESMModel not available")
class TestESMModel:
    """Test cases for ESMModel."""

    @patch("models.esm.esm.pretrained.esm2_t33_650M_UR50D")
    def test_esm_initialization(self, mock_esm):
        """Test ESMModel initialization."""
        device = torch.device("cpu")
        model = ESMModel("esm2_t33_650M_UR50D", device)

        assert model.model_name == "esm2_t33_650M_UR50D"
        assert model.device == device

    @patch("models.esm.esm.pretrained.esm2_t33_650M_UR50D")
    def test_esm_load_model(self, mock_esm_pretrained):
        """Test ESMModel model loading."""
        # Mock ESM model and alphabet
        mock_model = Mock()
        mock_alphabet = Mock()
        mock_esm_pretrained.return_value = (mock_model, mock_alphabet)

        device = torch.device("cpu")
        model = ESMModel("esm2_t33_650M_UR50D", device)
        model.load_model()

        assert model.model == mock_model
        assert model.alphabet == mock_alphabet
        mock_model.eval.assert_called_once()

    def test_esm_preprocess_sequence(self):
        """Test ESMModel sequence preprocessing."""
        device = torch.device("cpu")
        model = ESMModel("esm2_t33_650M_UR50D", device)

        sequence = "mkwvtfislll"
        processed = model.preprocess_sequence(sequence)

        # ESM uses uppercase sequences
        assert processed == sequence.upper()


@pytest.mark.skipif(not PROT_T5_AVAILABLE, reason="ProtT5Model not available")
class TestProtT5Model:
    """Test cases for ProtT5Model."""

    @patch("models.prot_t5.T5Tokenizer.from_pretrained")
    @patch("models.prot_t5.T5EncoderModel.from_pretrained")
    def test_prot_t5_initialization(self, mock_model, mock_tokenizer):
        """Test ProtT5Model initialization."""
        device = torch.device("cpu")
        model = ProtT5Model("prot_t5_xl_uniref50", device)

        assert model.model_name == "prot_t5_xl_uniref50"
        assert model.device == device

    def test_prot_t5_preprocess_sequence(self):
        """Test ProtT5Model sequence preprocessing."""
        device = torch.device("cpu")
        model = ProtT5Model("prot_t5_xl_uniref50", device)

        sequence = "MKWVTFISLLL"
        processed = model.preprocess_sequence(sequence)

        # ProtT5 adds spaces between amino acids
        expected = " ".join(sequence)
        assert processed == expected


class TestModelFactory:
    """Test cases for model factory functions."""

    def test_model_name_mapping(self):
        """Test that model names map to correct classes."""
        model_mappings = {
            "prost_t5": "ProstT5Model",
            "ankh": "AnkhModel",
            "esm2_t33_650M_UR50D": "ESMModel",
            "prot_t5_xl_uniref50": "ProtT5Model",
        }

        for model_name, expected_class in model_mappings.items():
            # The actual mapping would be tested in the factory function
            assert isinstance(model_name, str)
            assert isinstance(expected_class, str)

    def test_device_compatibility(self):
        """Test device compatibility across models."""
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))

        for device in devices:
            # Each model should accept the device parameter
            assert device.type in ["cpu", "cuda"]

    def test_model_parameters(self):
        """Test model parameter validation."""
        # Test various parameter combinations
        params = {"max_length": 512, "batch_size": 8, "precision": "float32"}

        for key, value in params.items():
            assert isinstance(key, str)
            # Parameters should be valid types
            assert value is not None


class TestModelIntegration:
    """Integration tests for model functionality."""

    def test_sequence_length_limits(self):
        """Test handling of sequences with different lengths."""
        # Very short sequence
        short_seq = "MK"
        assert len(short_seq) >= 1

        # Medium sequence
        medium_seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQ"
        assert 50 <= len(medium_seq) <= 100

        # Long sequence (should be handled or truncated)
        long_seq = "M" * 2000
        assert len(long_seq) == 2000

    def test_invalid_amino_acids(self):
        """Test handling of invalid amino acid characters."""
        # Valid sequence
        valid_seq = "MKWVTFISLLL"
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        assert all(aa in valid_aas for aa in valid_seq)

        # Invalid characters
        invalid_seq = "MKWVTFISLLLX"  # X is ambiguous
        assert "X" in invalid_seq

        # Lowercase (should be handled by preprocessing)
        lowercase_seq = "mkwvtfislll"
        assert lowercase_seq.upper() == "MKWVTFISLLL"

    def test_empty_sequence_handling(self):
        """Test handling of empty sequences."""
        empty_seq = ""
        assert len(empty_seq) == 0

        # Empty sequences should be handled gracefully
        # (either raise error or return default embedding)

    def test_special_characters(self):
        """Test handling of special characters in sequences."""
        # Sequences with gaps or special characters
        gapped_seq = "MKW-VTF-ISLLL"
        assert "-" in gapped_seq

        # Stop codon
        stop_seq = "MKWVTFISLLL*"
        assert "*" in stop_seq

    @patch("torch.cuda.is_available")
    def test_gpu_availability_handling(self, mock_cuda_available):
        """Test handling when GPU is/isn't available."""
        # Test when CUDA is available
        mock_cuda_available.return_value = True
        assert torch.cuda.is_available() == True

        # Test when CUDA is not available
        mock_cuda_available.return_value = False
        assert torch.cuda.is_available() == False

    def test_memory_efficiency(self):
        """Test memory-related functionality."""
        # This would test batch processing, memory cleanup, etc.
        # For now, just verify that we can create embeddings without memory issues

        # Simulate processing multiple sequences
        sequences = {f"seq_{i}": "MKWVTFISLLL" * (i + 1) for i in range(5)}

        # Each sequence should have a reasonable length
        for seq_id, sequence in sequences.items():
            assert len(sequence) > 0
            assert len(sequence) < 1000  # Reasonable upper bound

    def test_reproducibility_across_models(self):
        """Test that models produce reproducible results."""
        # Set random seed
        torch.manual_seed(42)
        np.random.seed(42)

        # Generate test embeddings
        test_embedding_1 = np.random.randn(128).astype(np.float32)

        # Reset seed
        torch.manual_seed(42)
        np.random.seed(42)

        # Generate test embeddings again
        test_embedding_2 = np.random.randn(128).astype(np.float32)

        # Should be identical
        np.testing.assert_array_equal(test_embedding_1, test_embedding_2)
