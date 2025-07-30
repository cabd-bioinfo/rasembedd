"""Tests for base model functionality."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from models.base_model import BaseEmbeddingModel


class MockEmbeddingModel(BaseEmbeddingModel):
    """Mock implementation of BaseEmbeddingModel for testing."""
    
    def load_model(self):
        """Mock model loading."""
        self.model = Mock()
        self.tokenizer = Mock()
    
    def preprocess_sequence(self, sequence: str) -> str:
        """Mock sequence preprocessing."""
        return sequence.upper()
    
    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Mock embedding generation."""
        # Return a deterministic embedding based on sequence length
        np.random.seed(len(sequence))
        return np.random.randn(128).astype(np.float32)


class TestBaseEmbeddingModel:
    """Test cases for BaseEmbeddingModel."""
    
    def test_init(self):
        """Test model initialization."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device, param1='value1')
        
        assert model.model_name == 'test_model'
        assert model.device == device
        assert model.model is None
        assert model.tokenizer is None
        assert model.model_kwargs == {'param1': 'value1'}
    
    def test_load_model(self):
        """Test model loading."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device)
        model.load_model()
        
        assert model.model is not None
        assert model.tokenizer is not None
    
    def test_preprocess_sequence(self):
        """Test sequence preprocessing."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device)
        
        result = model.preprocess_sequence('mkwvtfislll')
        assert result == 'MKWVTFISLLL'
    
    def test_generate_embedding(self):
        """Test single embedding generation."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device)
        
        embedding = model.generate_embedding('MKWVTFISLLL', 'test_seq')
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (128,)
        assert embedding.dtype == np.float32
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    def test_generate_embeddings_single(self, sample_sequences):
        """Test embedding generation for multiple sequences."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device)
        
        # Test with single sequence
        single_seq = {'test_seq_1': sample_sequences['test_seq_1']}
        embeddings = model.generate_embeddings(single_seq)
        
        assert len(embeddings) == 1
        assert 'test_seq_1' in embeddings
        assert isinstance(embeddings['test_seq_1'], np.ndarray)
    
    def test_generate_embeddings_multiple(self, sample_sequences):
        """Test embedding generation for multiple sequences."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device)
        
        embeddings = model.generate_embeddings(sample_sequences)
        
        assert len(embeddings) == len(sample_sequences)
        for seq_id in sample_sequences.keys():
            assert seq_id in embeddings
            assert isinstance(embeddings[seq_id], np.ndarray)
            assert embeddings[seq_id].shape == (128,)
    
    @patch('logging.getLogger')
    def test_generate_embeddings_with_error(self, mock_logger, sample_sequences):
        """Test embedding generation with error handling."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device)
        
        # Mock generate_embedding to raise an exception for one sequence
        original_method = model.generate_embedding
        def side_effect(sequence, seq_id):
            if seq_id == 'test_seq_2':
                raise ValueError("Test error")
            return original_method(sequence, seq_id)
        
        model.generate_embedding = Mock(side_effect=side_effect)
        
        embeddings = model.generate_embeddings(sample_sequences)
        
        # Should have embeddings for all sequences except the one that failed
        assert len(embeddings) == len(sample_sequences) - 1
        assert 'test_seq_1' in embeddings
        assert 'test_seq_3' in embeddings
        assert 'test_seq_2' not in embeddings
        
        # Check that error was logged
        mock_logger.return_value.error.assert_called()
    
    def test_reproducibility(self):
        """Test that embeddings are reproducible."""
        device = torch.device('cpu')
        model = MockEmbeddingModel('test_model', device)
        
        seq = 'MKWVTFISLLL'
        seq_id = 'test_seq'
        
        embedding1 = model.generate_embedding(seq, seq_id)
        embedding2 = model.generate_embedding(seq, seq_id)
        
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_device_assignment(self):
        """Test device assignment."""
        # Test CPU device
        cpu_device = torch.device('cpu')
        model_cpu = MockEmbeddingModel('test_model', cpu_device)
        assert model_cpu.device == cpu_device
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda:0')
            model_cuda = MockEmbeddingModel('test_model', cuda_device)
            assert model_cuda.device == cuda_device
