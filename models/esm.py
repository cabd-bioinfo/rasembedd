"""ESM (Evolutionary Scale Modeling) model implementation for protein embeddings."""

import re

import numpy as np
import torch

from .base_model import BaseEmbeddingModel


class ESMModel(BaseEmbeddingModel):
    def get_residue_embeddings(self, sequence: str, seq_id: str) -> np.ndarray:
        """Return per-residue embeddings (excluding special tokens). Shape: [seq_len, dim]"""
        processed_sequence = self.preprocess_sequence(sequence)
        encoded = self.tokenizer(
            processed_sequence,
            add_special_tokens=True,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state  # (batch, seq_len, dim)
            # Remove batch dimension and special tokens (assume 1st and last are special)
            valid_len = int(attention_mask.sum().item())
            residue_embeddings = token_embeddings[0, 1 : valid_len - 1, :]
            return residue_embeddings.cpu().numpy()

    def get_mean_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Return mean-pooled embedding (shape: [dim])."""
        processed_sequence = self.preprocess_sequence(sequence)
        encoded = self.tokenizer(
            processed_sequence,
            add_special_tokens=True,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            masked_embeddings = token_embeddings * mask_expanded
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            return mean_embedding.cpu().numpy().squeeze()

    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Legacy interface: returns mean embedding (for backward compatibility)."""
        return self.get_mean_embedding(sequence, seq_id)

    """ESM model implementation using transformers library."""

    def __init__(self, model_name: str = None, device: torch.device = None, **kwargs):
        # Default model name if not provided
        if model_name is None:
            model_name = "facebook/esm2_t33_650M_UR50D"  # ESM-2 medium model

        super().__init__(model_name, device, **kwargs)

        # ESM-specific parameters
        self.max_length = kwargs.get("max_length", 1024)
        self.pooling_strategy = kwargs.get("pooling_strategy", "mean")  # 'mean', 'cls', 'max'

    def load_model(self) -> None:
        """Load ESM model and tokenizer."""
        try:
            from transformers import EsmModel, EsmTokenizer

            self.tokenizer = EsmTokenizer.from_pretrained(self.model_name)
            self.model = EsmModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()

        except ImportError:
            raise ImportError(
                "transformers library is required for ESM models. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ESM model '{self.model_name}': {str(e)}")

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for ESM."""
        # ESM handles standard amino acids, replace unusual ones with X
        cleaned_sequence = re.sub(r"[UZOB]", "X", sequence.upper())
        return cleaned_sequence  # ESM doesn't need spaces between amino acids

    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Generate embedding for a single sequence using ESM."""
        # Preprocess sequence
        processed_sequence = self.preprocess_sequence(sequence)

        # Tokenize sequence
        encoded = self.tokenizer(
            processed_sequence,
            add_special_tokens=True,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            if self.pooling_strategy == "cls":
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :]  # CLS token is at position 0
            elif self.pooling_strategy == "mean":
                # Apply mean pooling with attention mask
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                masked_embeddings = token_embeddings * mask_expanded
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            elif self.pooling_strategy == "max":
                # Apply max pooling
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                masked_embeddings = token_embeddings * mask_expanded
                # Set masked positions to large negative value for max pooling
                masked_embeddings[mask_expanded == 0] = -1e9
                embedding = torch.max(masked_embeddings, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

            return embedding.cpu().numpy().squeeze()
