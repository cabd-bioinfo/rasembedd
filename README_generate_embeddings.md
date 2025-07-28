# generate_embeddings.py Documentation

## Overview
`generate_embeddings.py` is a unified script for generating protein embeddings using various protein language models. It supports multiple models, flexible input/output formats, and is designed for extensibility and reproducibility.

---

## Requirements
- Python 3.7+
- PyTorch
- pandas
- numpy
- argparse
- pickle (standard library)
- h5py (optional, for HDF5 output)

Install requirements (example):
```bash
pip install torch pandas numpy h5py
```

---

## Usage

### Basic Command
```bash
python generate_embeddings.py --input data/RAS.fasta --input_type fasta --model_type prot_t5 --output embeddings.pkl
```

### Arguments
- `--input` (required): Path to input file (FASTA or TSV)
- `--input_type`: Type of input file (`fasta` or `tsv`, default: `tsv`)
- `--id_column`: Column name for sequence IDs (TSV only, default: `uniprot_id`)
- `--seq_column`: Column name for sequences (TSV only, default: `sequence`)
- `--model_type` (required): Model to use (e.g., `prot_t5`, `esm`, `custom`)
- `--model_name`: Specific model name/path (model-specific)
- `--device`: Device to use (`auto`, `cuda`, or `cpu`, default: `auto`)
- `--output` (required): Path to output file for embeddings
- `--format`: Output format (`pickle`, `npz`, `hdf5`, `h5`, default: `pickle`)
- `--model_args`: Additional model-specific arguments (key=value pairs)

---

## Output Formats
- `pickle`: Python pickle file (default)
- `npz`: Numpy compressed file
- `hdf5`/`h5`: HDF5 file (requires `h5py`)

---

## Examples

### 1. Generate embeddings from a FASTA file using ProtT5
```bash
python generate_embeddings.py --input data/RAS.fasta --input_type fasta --model_type prot_t5 --output embeddings.pkl
```

### 2. Generate embeddings from a TSV file using ESM model, output as HDF5
```bash
python generate_embeddings.py --input data/RAS.updated.Info.Table.V7.tsv --input_type tsv --model_type esm --output embeddings.h5 --format hdf5
```

### 3. Specify device and model arguments
```bash
python generate_embeddings.py --input data/RAS.fasta --input_type fasta --model_type custom --model_name my_model.pt --device cuda --output embeddings.npz --format npz --model_args batch_size=16
```

---

## Notes
- The script will automatically detect and use available CUDA devices if `--device auto` is set.
- Model-specific arguments can be passed using `--model_args key=value`.
- The script supports dynamic loading of models from the `models/` directory.
- For HDF5 output, install `h5py`.

---

## Troubleshooting
- If you see `Model type 'xyz' not found`, check the `models/` directory for available models.
- For HDF5 errors, ensure `h5py` is installed.
- For CUDA errors, check your PyTorch and CUDA installation.

---

## License
MIT or as specified in the repository.
