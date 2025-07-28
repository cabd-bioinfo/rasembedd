"""Base class for protein embedding models."""

import torch
import numpy as np
from typing import Dict
from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    """Abstract base class for protein embedding models."""
    
    def __init__(self, model_name: str, device: torch.device, **kwargs):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Store additional model-specific parameters
        self.model_kwargs = kwargs
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess a protein sequence for the specific model."""
        pass
    
    @abstractmethod
    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Generate embedding for a single sequence."""
        pass
    
    def generate_embeddings(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Generate embeddings for multiple sequences."""
        import logging
        logger = logging.getLogger(__name__)
        
        embeddings = {}
        total_sequences = len(sequences)
        
        for i, (seq_id, sequence) in enumerate(sequences.items()):
            logger.info(f"Processing sequence {i+1}/{total_sequences}: {seq_id}")
            
            try:
                embedding = self.generate_embedding(sequence, seq_id)
                embeddings[seq_id] = embedding
            except Exception as e:
                logger.error(f"Error processing sequence {seq_id}: {str(e)}")
                continue
        
        logger.info(f"Successfully generated embeddings for {len(embeddings)} sequences")
        return embeddings
