# Building a Custom Model Module for Protein Embeddings

## Overview
This guide explains how to create a new model module for use with `generate_embeddings.py`. Each model module should implement a class that inherits from `BaseEmbeddingModel` (found in `models/base_model.py`). This allows the script to dynamically load and use your model for embedding generation.

---

## Requirements
- Python 3.7+
- Your model's dependencies (e.g., PyTorch, transformers, etc.)
- The `models/` directory must contain your module as a `.py` file

---

## Steps to Build a Model Module

### 1. Inherit from `BaseEmbeddingModel`
Your model class must inherit from `BaseEmbeddingModel` and implement the required methods:
- `__init__(self, model_name=None, device='cpu', **kwargs)`
- `load_model(self)`
- `generate_embeddings(self, sequences: Dict[str, str]) -> Dict[str, np.ndarray]`

### 2. Save Your Module
- Place your module as a `.py` file in the `models/` directory (e.g., `models/my_model.py`).
- The filename (without `.py`) will be the model type used in the `--model_type` argument.

### 3. Example Skeleton
```python
from models.base_model import BaseEmbeddingModel
import numpy as np

class MyModel(BaseEmbeddingModel):
    def __init__(self, model_name=None, device='cpu', **kwargs):
        super().__init__(model_name, device, **kwargs)
        # Initialize your model here

    def load_model(self):
        # Load model weights, tokenizer, etc.
        pass

    def generate_embeddings(self, sequences):
        # sequences: Dict[str, str] (id -> sequence)
        embeddings = {}
        for seq_id, seq in sequences.items():
            # Generate embedding for seq (replace with your logic)
            embeddings[seq_id] = np.random.rand(1024)  # Example: random vector
        return embeddings
```

### 4. Model Discovery
- The script will automatically discover all `.py` files in `models/` (except `base_model.py` and files starting with `__`).
- The class name should end with `Model` and inherit from `BaseEmbeddingModel`.

### 5. Using Your Model
Run the embedding script with your model type:
```bash
python generate_embeddings.py --input data/RAS.fasta --input_type fasta --model_type my_model --output embeddings.pkl
```

---

## Tips
- You can add model-specific arguments by accepting `**kwargs` in your class and parsing them as needed.
- Use the `device` argument to support CPU/GPU selection.
- For large models, consider batch processing in `generate_embeddings`.

---

## Troubleshooting
- If your model is not found, check the filename and class name.
- Ensure your class inherits from `BaseEmbeddingModel`.
- Catch and log exceptions in `load_model` and `generate_embeddings` for easier debugging.

---

## License
MIT or as specified in the repository.
