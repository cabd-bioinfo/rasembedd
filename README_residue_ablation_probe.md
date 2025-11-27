# residue_ablation_probe.py Documentation

## Overview
`residue_ablation_probe.py` performs residue-level ablation analysis for protein cluster assignment interpretation. It perturbs protein sequences, applies a protein language model (PLM) and a trained linear probe, and quantifies the impact of each residue on cluster assignment. Outputs include per-residue ablation scores and visualizations.

---

## Requirements
- Python 3.7+
- pandas
- numpy
- torch
- matplotlib
- seaborn
- biopython

Install requirements (example):
```bash
pip install pandas numpy torch matplotlib seaborn biopython
```

---

## Usage

### Basic Command
```bash
python residue_ablation_probe.py --input proteins.tsv --model_type prot_t5 --probe_model probe.pkl --output_prefix results/ablation
```

### Arguments

#### Input & Model
- `--input` (required): Input protein file (TSV or FASTA)
- `--input_type`: Input file type (`tsv` or `fasta`, default: `tsv`)
- `--id_column`: ID column for TSV input (default: `uniprot_id`)
- `--seq_column`: Sequence column for TSV input (default: `sequence`)
- `--model_type` (required): PLM model type (as in generate_embeddings.py)
- `--model_name`: Model name/path (model-specific)
- `--device`: Device to use (`auto`, `cuda`, `cpu`, default: `auto`)
- `--probe_model` (required): Trained linear probe model (pickle)
- `--model_args`: Additional model-specific arguments (key=value pairs)

#### Ablation & Scanning
- `--window_size`: Window size L for ablation (default: 5; ignored for `--scan_mode ala-scanning`)
- `--scan_mode`: Scanning mode (`ablation` or `ala-scanning`, default: `ablation`)
- `--mutation_policy`: Replacement policy for `ala-scanning` (`alanine`, `to-x`, `random`, `blosum`, default: `alanine`)
- `--random_seed`: Random seed for mutation policies (default: 42)
- `--blosum_name`: BLOSUM matrix for mutation policy (`blosum62`, default: `blosum62`)
- `--blosum_temp`: Temperature for BLOSUM softmax sampling (default: 1.0)

#### Output
- `--output_prefix` (required): Prefix for output files

#### Performance & Parallelization
- `--n-processes`: Number of parallel threads to use for protein processing (default: 1 for serial processing)

#### Plotting & Visualization
- `--no-progress`: Disable progress indicators
- `--no-heatmap`: Skip cluster heatmap plots
- `--no-alignment-plots`: Skip MSA/alignment plots
- `--no-plots`: Skip all plotting/visualizations
- `--max-clusters-to-plot`: Max clusters to plot (default: all)
- `--max-msa-seqs`: Max sequences in MSA before skipping colored MSA plot (default: 100)
- `--max-msa-length`: Max MSA alignment length before skipping colored MSA plot (default: 2000)
- `--force-plots`: Force plot generation even if exceeding safety thresholds
- `--subsample-msa`: Subsample to `--max-msa-seqs` for MSA/plotting if exceeded
- `--max-total-residues`: Max total residues for MSA plotting (default: 50000)

#### Normalization
- `--no-standardize`: Skip feature standardization
- `--norm-center`: Center features (subtract mean, default: True)
- `--no-norm-center`: Don't center features
- `--norm-scale`: Scale features (divide by std, default: True)
- `--no-norm-scale`: Don't scale features
- `--norm-pca-components`: PCA components or variance to retain (default: 0.95)
- `--norm-l2`: Apply L2 normalization (default: True)
- `--no-norm-l2`: Don't apply L2 normalization

#### SDP Annotation
- `--sdp_tsv`: Optional path to SDP TSV file (first column: protein ids, next columns: residue coordinates)

---

## Output Format
- Per-residue ablation scores (TSV)
- Cluster heatmaps (PDF/PNG, if enabled)
- Colored MSA plots (PDF/PNG, if enabled)

---

## Example
```bash
# Basic ablation analysis
python residue_ablation_probe.py --input proteins.tsv --model_type prot_t5 --probe_model probe.pkl --output_prefix results/ablation --window_size 7 --scan_mode ablation --no-progress

# Parallel processing with 16 threads for faster analysis
python residue_ablation_probe.py --input proteins.tsv --model_type prot_t5 --probe_model probe.pkl --output_prefix results/ablation --n-processes 16 --scan_mode ablation
```

---

## Notes
- Supports both ablation and alanine scanning modes.
- Multiple mutation policies for residue replacement.
- Parallel processing available using `--n-processes` for faster analysis on multi-core systems.
- Extensive plotting and normalization options.
- Compatible with outputs from generate_embeddings.py and linear_probe.py.

---

## Troubleshooting
- For model loading errors, check model path and type.
- For device errors, ensure CUDA is available if using GPU.
- For input format errors, verify file and column names.
- For plotting errors, check matplotlib and seaborn installation.

---

## License
MIT or as specified in the repository.
- `--ablation_value`: Value to use for ablation (e.g., `X` for masking)
- `--save_intermediate`: Save intermediate results (default: False)

---

## Output Format
- Tab-separated file (`.tsv`) with columns:
  - `protein_id`: Sequence identifier
  - `residue`: Residue position
  - `aa`: Original amino acid
  - `ablation_score`: Impact score for each residue
  - Additional columns as needed (e.g., model output, class probabilities)

---

## Examples

### 1. Run ablation analysis on a FASTA file
```bash
python residue_ablation_probe.py --input data/RAS.fasta --input_type fasta --model model.pt --output ablation_scores.tsv
```

### 2. Use a custom ablation method and batch size
```bash
python residue_ablation_probe.py --input data/RAS.tsv --input_type tsv --model model.pt --ablation_method mask --batch_size 8 --output ablation_scores.tsv
```

---

## Notes
- The script supports both FASTA and TSV input formats.
- Ablation scores quantify the change in model output when each residue is perturbed.
- For large datasets, use a higher batch size for faster inference if memory allows.
- Model-specific arguments may be required depending on your setup.

---

## Troubleshooting
- If you see errors related to model loading, check your model path and type.
- For device errors, ensure your environment supports CUDA if using GPU.
- For input format errors, verify your file and column names.

---

## License
MIT or as specified in the repository.
