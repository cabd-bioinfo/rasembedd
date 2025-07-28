"""ProtT5 model implementation for protein embeddings."""

import torch
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
import re
from .base_model import BaseEmbeddingModel

class ProtT5Model(BaseEmbeddingModel):
    """ProtT5 model implementation."""
    
    def __init__(self, model_name: str = None, device: torch.device = None, **kwargs):
        # Default model name if not provided
        if model_name is None:
            model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
        
        super().__init__(model_name, device, **kwargs)
        
        # ProtT5-specific parameters
        self.max_length = kwargs.get('max_length', None)
        self.use_half_precision = kwargs.get('use_half_precision', False)
    
    def load_model(self) -> None:
        """Load ProtT5 model and tokenizer."""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(self.model_name)
            
            if self.use_half_precision:
                self.model = self.model.half()
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ProtT5 model '{self.model_name}': {str(e)}")
    
    def preprocess_sequence(self, sequence: str) -> str:
        """Preprocess protein sequence for ProtT5."""
        # Replace unusual amino acids (U, Z, O, B) with X
        cleaned_sequence = re.sub(r"[UZOB]", "X", sequence.upper())
        # Add spaces between amino acids for tokenization
        return " ".join(list(cleaned_sequence))
    
    def generate_embedding(self, sequence: str, seq_id: str) -> np.ndarray:
        """Generate embedding for a single sequence using ProtT5."""
        # Preprocess sequence
        processed_sequence = self.preprocess_sequence(sequence)
        
        # Determine max_length
        if self.max_length is None:
            # Use sequence length + buffer for special tokens
            max_len = min(len(sequence) + 2, self.tokenizer.model_max_length)
        else:
            max_len = min(self.max_length, self.tokenizer.model_max_length)
        
        # Tokenize sequence
        encoded = self.tokenizer(
            processed_sequence,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Apply mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state
            
            # Expand attention mask to match embedding dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            
            # Apply mask and compute mean
            masked_embeddings = token_embeddings * mask_expanded
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embedding = sum_embeddings / sum_mask
            
            return mean_embedding.cpu().numpy().squeeze()
