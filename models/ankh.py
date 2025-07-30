"""Ankh model implementation for protein embeddings."""

import re

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from .base_model import BaseEmbeddingModel


class AnkhModel(BaseEmbeddingModel):
    """Ankh model implementation using T5 architecture."""

    def __init__(self, model_name: str = None, device: torch.device = None, **kwargs):
        # Default model name if not provided
        if model_name is None:
            model_name = "ElnaggarLab/ankh3-large"

        super().__init__(model_name, device, **kwargs)

        # Ankh-specific parameters
        self.max_length = kwargs.get("max_length", None)
        self.use_half_precision = kwargs.get("use_half_precision", False)
        self.prefix = kwargs.get("prefix", "[NLU]")  # Options: '[NLU]', '[S2S]'
        self.pooling_strategy = kwargs.get("pooling_strategy", "mean")  # 'mean', 'cls', 'max'

    def load_model(self) -> None:
        """Load Ankh model and tokenizer."""
        try:
            # Must use T5Tokenizer, not AutoTokenizer for Ankh models
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5EncoderModel.from_pretrained(self.model_name)

            if self.use_half_precision:
                self.model = self.model.half()

            self.model = self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load Ankh model '{self.model_name}': {str(e)}")

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for Ankh."""
        # Clean the sequence
        cleaned_sequence = sequence.upper().strip()

        # Replace unusual amino acids with standard ones
        cleaned_sequence = re.sub(r"[UZOB]", "X", cleaned_sequence)

        # Add the prefix - Ankh models use special prefixes
        # [NLU] for natural language understanding tasks (general embeddings)
        # [S2S] for sequence-to-sequence tasks (might give better embeddings for some tasks)
        prefixed_sequence = self.prefix + cleaned_sequence

        return prefixed_sequence

    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Generate embedding for a single sequence using Ankh."""
        # Preprocess sequence
        processed_sequence = self.preprocess_sequence(sequence)

        # Determine max_length
        if self.max_length is None:
            # Use sequence length + buffer for special tokens and prefix
            max_len = min(len(processed_sequence) + 10, self.tokenizer.model_max_length)
        else:
            max_len = min(self.max_length, self.tokenizer.model_max_length)

        # Tokenize sequence
        # Important: use is_split_into_words=False for Ankh models
        encoded = self.tokenizer(
            processed_sequence,
            add_special_tokens=True,
            return_tensors="pt",
            is_split_into_words=False,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
        )

        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Get token embeddings
            token_embeddings = (
                outputs.last_hidden_state
            )  # Shape: (batch_size, seq_len, hidden_size)

            # Apply pooling strategy
            if self.pooling_strategy == "cls":
                # Use the first token (CLS-like token)
                embedding = token_embeddings[:, 0, :]
            elif self.pooling_strategy == "mean":
                # Apply mean pooling with attention mask
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                masked_embeddings = token_embeddings * mask_expanded
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embedding = sum_embeddings / sum_mask
            elif self.pooling_strategy == "max":
                # Apply max pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                masked_embeddings = token_embeddings * mask_expanded
                # Set masked positions to large negative value for max pooling
                masked_embeddings[mask_expanded == 0] = -1e9
                embedding = torch.max(masked_embeddings, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

            return embedding.cpu().numpy().squeeze()

    @staticmethod
    def get_available_models():
        """Return list of available Ankh model variants."""
        return [
            "ElnaggarLab/ankh3-large",
            "ElnaggarLab/ankh3-base",
            "ElnaggarLab/ankh2",
            "ElnaggarLab/ankh1",
        ]

    @staticmethod
    def get_model_info():
        """Return information about Ankh models."""
        return {
            "description": "Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling",
            "paper": "https://arxiv.org/abs/2301.06568",
            "prefixes": {
                "[NLU]": "Natural Language Understanding - for general embeddings",
                "[S2S]": "Sequence-to-Sequence - may provide better embeddings for some tasks",
            },
            "recommended_settings": {
                "prefix": "[NLU]",
                "pooling_strategy": "mean",
                "max_length": 1024,
            },
        }
