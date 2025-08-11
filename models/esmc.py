"""ESMC model implementation for protein embeddings using ESM's native SDK."""

import re

import numpy as np
import torch

from .base_model import BaseEmbeddingModel


class ESMCModel(BaseEmbeddingModel):
    def get_residue_embeddings(self, sequence: str, seq_id: str) -> np.ndarray:
        """Return per-residue embeddings (excluding special tokens). Shape: [seq_len, dim]"""
        processed_sequence = self.preprocess_sequence(sequence)
        protein = self.ESMProtein(sequence=processed_sequence)
        protein_tensor = self.model.encode(protein)
        logits_config = self.LogitsConfig(
            sequence=False,  # Return token-level embeddings
            return_embeddings=True,
            return_hidden_states=True,
            ith_hidden_layer=self.layer_idx,
        )
        logits_output = self.model.logits(protein_tensor, logits_config)
        if logits_output.hidden_states is not None and self.layer_idx != -1:
            embeddings = logits_output.hidden_states
            if embeddings.dim() == 4:
                if embeddings.shape[0] == 1:
                    embeddings = embeddings.squeeze(0)
                else:
                    embeddings = embeddings[0]
        elif logits_output.embeddings is not None:
            embeddings = logits_output.embeddings
        else:
            raise RuntimeError("No embeddings returned from ESMC model")
        # Remove batch dimension if present
        if embeddings.dim() == 3:
            residue_embeddings = embeddings[0]
        else:
            residue_embeddings = embeddings
        return residue_embeddings.detach().cpu().numpy()

    def get_mean_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Return mean-pooled embedding (shape: [dim])."""
        processed_sequence = self.preprocess_sequence(sequence)
        protein = self.ESMProtein(sequence=processed_sequence)
        protein_tensor = self.model.encode(protein)
        logits_config = self.LogitsConfig(
            sequence=False,  # Return token-level embeddings
            return_embeddings=True,
            return_hidden_states=True,
            ith_hidden_layer=self.layer_idx,
        )
        logits_output = self.model.logits(protein_tensor, logits_config)
        if logits_output.hidden_states is not None and self.layer_idx != -1:
            embeddings = logits_output.hidden_states
            if embeddings.dim() == 4:
                if embeddings.shape[0] == 1:
                    embeddings = embeddings.squeeze(0)
                else:
                    embeddings = embeddings[0]
        elif logits_output.embeddings is not None:
            embeddings = logits_output.embeddings
        else:
            raise RuntimeError("No embeddings returned from ESMC model")
        # Mean pool over sequence length (dim=1)
        if embeddings.dim() == 3:
            embedding = torch.mean(embeddings, dim=1)
        else:
            embedding = embeddings
        return embedding.detach().cpu().numpy().squeeze()

    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Legacy interface: returns mean embedding (for backward compatibility)."""
        return self.get_mean_embedding(sequence, seq_id)

    """ESMC model implementation using ESM's native SDK."""

    def __init__(self, model_name: str = None, device: torch.device = None, **kwargs):
        # Default model name if not provided
        if model_name is None:
            model_name = "esmc_300m"

        super().__init__(model_name, device, **kwargs)

        # ESMC-specific parameters
        self.return_logits = kwargs.get("return_logits", False)
        self.pooling_strategy = kwargs.get("pooling_strategy", "mean")  # 'mean', 'cls', 'max'
        self.sequence_level = kwargs.get("sequence_level", True)  # Return sequence-level embeddings
        self.layer_idx = kwargs.get("layer_idx", -1)  # Specific layer index (-1 for all layers)

        # Store ESM classes
        self.ESMC = None
        self.ESMProtein = None
        self.LogitsConfig = None

    def load_model(self) -> None:
        """Load ESMC model using ESM's native SDK."""
        try:
            # Import ESM modules
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig

            self.ESMC = ESMC
            self.ESMProtein = ESMProtein
            self.LogitsConfig = LogitsConfig

            # Load the model
            device_str = "cuda" if self.device.type == "cuda" else "cpu"
            self.model = self.ESMC.from_pretrained(self.model_name).to(device_str)

        except ImportError as e:
            raise ImportError(
                f"ESM SDK is required for ESMC models. Install with: pip install fair-esm\n"
                f"Original error: {str(e)}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ESMC model '{self.model_name}': {str(e)}")

    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for ESMC."""
        # Clean the sequence
        cleaned_sequence = sequence.upper().strip()

        # Replace unusual amino acids with standard ones
        # ESMC typically handles standard amino acids well
        cleaned_sequence = re.sub(r"[UZOB]", "X", cleaned_sequence)

        return cleaned_sequence

    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Generate embedding for a single sequence using ESMC."""
        # Preprocess sequence
        processed_sequence = self.preprocess_sequence(sequence)

        # Create ESM protein object
        protein = self.ESMProtein(sequence=processed_sequence)

        # Encode the protein
        protein_tensor = self.model.encode(protein)

        # Get embeddings using logits output with layer specification
        logits_config = self.LogitsConfig(
            sequence=self.sequence_level,
            return_embeddings=True,
            return_hidden_states=True,
            ith_hidden_layer=self.layer_idx,  # Specify which layer to extract
        )

        logits_output = self.model.logits(protein_tensor, logits_config)

        # Extract embeddings - prefer hidden_states if available and layer is specified
        if logits_output.hidden_states is not None and self.layer_idx != -1:
            # Use specific layer from hidden states
            embeddings = logits_output.hidden_states
            if (
                embeddings.dim() == 4
            ):  # [1, seq_len, hidden_dim] or [n_layers, batch, seq_len, hidden_dim]
                if embeddings.shape[0] == 1:  # Single layer returned
                    embeddings = embeddings.squeeze(0)  # Remove layer dimension
                else:
                    embeddings = embeddings[0]  # Take first (and likely only) layer
        elif logits_output.embeddings is not None:
            # Use regular embeddings
            embeddings = logits_output.embeddings
        else:
            raise RuntimeError("No embeddings returned from ESMC model")

        # Apply pooling strategy if we have token-level embeddings
        if embeddings.dim() > 2:  # (batch, seq_len, hidden_dim)
            if self.pooling_strategy == "mean":
                # Mean pooling across sequence length
                embedding = torch.mean(embeddings, dim=1)
            elif self.pooling_strategy == "cls":
                # Use first token (CLS-like)
                embedding = embeddings[:, 0, :]
            elif self.pooling_strategy == "max":
                # Max pooling across sequence length
                embedding = torch.max(embeddings, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        else:
            # Already pooled embeddings
            embedding = embeddings

        # Convert to numpy and squeeze batch dimension
        return embedding.detach().cpu().numpy().squeeze()

    @staticmethod
    def get_available_models():
        """Return list of available ESMC model variants."""
        return [
            "esmc_300m",
            "esmc_600m",
            # Add other ESMC variants as they become available
        ]

    @staticmethod
    def get_model_info():
        """Return information about ESMC models."""
        return {
            "description": "ESM-C: Evolutionary-scale prediction of protein function with ESM-C",
            "paper": "Check ESM repository for latest papers",
            "sdk": "Uses ESM native SDK (esm)",
            "installation": "pip install esm",
            "features": {
                "sequence_level": "Return sequence-level embeddings",
                "return_logits": "Also return logits if needed",
                "pooling_strategy": "How to pool token-level embeddings",
                "layer_idx": "Specific layer to extract embeddings from (-1 for all layers)",
            },
            "recommended_settings": {
                "pooling_strategy": "mean",
                "sequence_level": True,
                "return_logits": False,
                "layer_idx": -1,
            },
            "layer_info": {
                "esmc_300m": "31 layers (0-30)",
                "esmc_600m": "Check model documentation for layer count",
                "layer_selection": "Use layer_idx parameter to select specific layer. Layer 12 often good for structural tasks, layer 30 for final representations.",
            },
        }

    @staticmethod
    def get_installation_instructions():
        """Return installation instructions for ESM SDK."""
        return """
To use ESMC models, you need to install the ESM SDK:

1. Install esm:
   pip install esm

2. Or install from source:
   git clone https://github.com/evolutionaryscale/esm.git
   cd esm
   pip install -e .

3. You may also need to install additional dependencies:
   pip install biotite

Note: The ESM SDK might have specific CUDA/PyTorch version requirements.
Check the official ESM repository for the latest installation instructions.

Layer Selection Guide:
- Use layer_idx=-1 to get embeddings from all layers (default)
- Use layer_idx=12 for good structural/functional representations
- Use layer_idx=30 for final layer representations (esmc_300m has 31 layers: 0-30)
- Different layers may work better for different downstream tasks
"""
