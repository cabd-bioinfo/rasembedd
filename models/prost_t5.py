"""ProstT5 model implementation for protein embeddings.

ProstT5 can process both amino acid sequences and 3Di structure sequences.
- For amino acid sequences: use <AA2fold> prefix
- For 3Di sequences: use <fold2AA> prefix
"""

import re

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from .base_model import BaseEmbeddingModel


class ProstT5Model(BaseEmbeddingModel):
    """ProstT5 model implementation for amino acid and 3Di structure sequences."""

    def __init__(self, model_name: str = None, device: torch.device = None, **kwargs):
        # Default model name if not provided
        if model_name is None:
            model_name = "Rostlab/ProstT5"

        super().__init__(model_name, device, **kwargs)

        # ProstT5-specific parameters
        self.max_length = kwargs.get("max_length", None)
        self.use_half_precision = kwargs.get(
            "use_half_precision", True
        )  # Default to half precision for GPU
        self.sequence_type = kwargs.get("sequence_type", "amino_acid")  # 'amino_acid' or '3di'
        self.layer_idx = kwargs.get(
            "layer_idx", -1
        )  # Layer to extract embeddings from (-1 for last layer)
        self.return_all_layers = kwargs.get(
            "return_all_layers", False
        )  # Return embeddings from all layers

    def load_model(self) -> None:
        """Load ProstT5 model and tokenizer."""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(self.model_name)

            # Apply precision settings
            if self.device.type == "cpu":
                self.model.float()
            else:
                if self.use_half_precision:
                    self.model.half()
                else:
                    self.model.float()

            self.model = self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load ProstT5 model '{self.model_name}': {str(e)}")

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for ProstT5.

        Args:
            sequence: Raw protein sequence (amino acids or 3Di)

        Returns:
            Preprocessed sequence with appropriate prefix
        """
        # Determine sequence type and apply appropriate preprocessing
        if self.sequence_type == "3di" or sequence.islower():
            # 3Di sequence (should be lowercase)
            processed = sequence.lower()
            prefix = "<fold2AA>"
        else:
            # Amino acid sequence (should be uppercase)
            # Replace unusual amino acids with X
            processed = re.sub(r"[UZOB]", "X", sequence.upper())
            prefix = "<AA2fold>"

        # Add spaces between characters
        spaced_sequence = " ".join(list(processed))

        # Add appropriate prefix
        return f"{prefix} {spaced_sequence}"

    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Generate embedding for a single sequence using ProstT5.

        Args:
            sequence: Protein sequence (amino acids or 3Di)
            seq_id: Sequence identifier

        Returns:
            Embedding vector as numpy array
        """
        # Preprocess sequence
        processed_sequence = self.preprocess_sequence(sequence)

        # Determine max_length
        if self.max_length is None:
            # Use sequence length + buffer for special tokens and prefix
            max_len = min(len(sequence) + 10, self.tokenizer.model_max_length)
        else:
            max_len = min(self.max_length, self.tokenizer.model_max_length)

        # Tokenize sequence
        encoded = self.tokenizer(
            processed_sequence,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Generate embeddings
        with torch.no_grad():
            # Get outputs with hidden states if needed
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.return_all_layers or self.layer_idx != -1,
            )

            # Select the appropriate layer
            if self.return_all_layers:
                # Return embeddings from all layers
                hidden_states = outputs.hidden_states
                # Stack all layers (layer, batch, seq_len, hidden_size)
                all_embeddings = torch.stack(hidden_states, dim=0)

                # Apply mean pooling for each layer
                embeddings_list = []
                for layer_emb in all_embeddings:
                    # Apply attention mask and mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(layer_emb.size())
                    masked_embeddings = layer_emb * mask_expanded
                    sum_embeddings = torch.sum(masked_embeddings, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    mean_embedding = sum_embeddings / sum_mask
                    embeddings_list.append(mean_embedding.cpu().numpy().squeeze())

                return np.array(embeddings_list)

            elif self.layer_idx != -1:
                # Extract from specific layer
                if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                    if self.layer_idx < len(outputs.hidden_states):
                        token_embeddings = outputs.hidden_states[self.layer_idx]
                    else:
                        print(f"Warning: Layer {self.layer_idx} not available, using last layer")
                        token_embeddings = outputs.last_hidden_state
                else:
                    token_embeddings = outputs.last_hidden_state
            else:
                # Use last layer (default)
                token_embeddings = outputs.last_hidden_state

            # Apply mean pooling with attention mask
            # Expand attention mask to match embedding dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())

            # Apply mask and compute mean (excluding special tokens and padding)
            # Remove special tokens: [CLS] at start and [SEP] at end, plus prefix token
            # For ProstT5, we typically want to exclude the first few tokens (special + prefix)
            actual_seq_start = 2  # Skip [CLS] and prefix token
            seq_length = attention_mask.sum().item() - 1  # Exclude [SEP]
            actual_seq_end = min(seq_length, token_embeddings.size(1))

            if actual_seq_end > actual_seq_start:
                # Extract embeddings for actual sequence (excluding prefix and special tokens)
                seq_embeddings = token_embeddings[:, actual_seq_start:actual_seq_end, :]
                seq_mask = attention_mask[:, actual_seq_start:actual_seq_end]

                # Apply mean pooling
                mask_expanded_seq = seq_mask.unsqueeze(-1).expand(seq_embeddings.size())
                masked_embeddings = seq_embeddings * mask_expanded_seq
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                sum_mask = torch.clamp(mask_expanded_seq.sum(dim=1), min=1e-9)
                mean_embedding = sum_embeddings / sum_mask
            else:
                # Fallback: use all tokens with masking
                masked_embeddings = token_embeddings * mask_expanded
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                mean_embedding = sum_embeddings / sum_mask

            return mean_embedding.cpu().numpy().squeeze()

    @staticmethod
    def get_model_info() -> dict:
        """Get information about ProstT5 model."""
        return {
            "name": "ProstT5",
            "description": "Protein Language Model for both amino acid sequences and 3Di structures",
            "paper": "ProstT5: Bilingual Language Model for Protein Sequence and Structure",
            "default_model": "Rostlab/ProstT5",
            "sequence_types": ["amino_acid", "3di"],
            "max_sequence_length": 1024,
            "embedding_dim": 1024,
            "layers": 24,
            "layer_info": {
                "total_layers": 24,
                "layer_range": "0-23 (0=first layer, 23=last layer)",
                "default": -1,  # Last layer
                "description": "Layer index for embedding extraction. -1 uses the final layer output.",
            },
            "prefixes": {"amino_acid": "<AA2fold>", "3di": "<fold2AA>"},
            "special_features": [
                "Bilingual: handles both amino acid and 3Di structure sequences",
                "Directional prefixes for sequence type specification",
                "Optimized for structure-sequence relationships",
            ],
        }

    @staticmethod
    def get_available_models() -> list:
        """Get list of available ProstT5 models."""
        return ["Rostlab/ProstT5"]
