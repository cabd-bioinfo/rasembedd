"""Base class for protein embedding models."""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch


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
    def get_residue_embeddings(self, sequence: str, seq_id: str) -> np.ndarray:
        """Return per-residue embeddings for a single sequence (shape: [seq_len, dim])."""
        pass

    @abstractmethod
    def get_mean_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Return mean-pooled embedding for a single sequence (shape: [dim])."""
        pass

    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Legacy interface: returns mean embedding (for backward compatibility)."""
        return self.get_mean_embedding(sequence, seq_id)

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
