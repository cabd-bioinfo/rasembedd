# Protein Embedding Visualization and Analysis Toolkit

This toolkit provides a comprehensive pipeline for protein embedding analysis, including:
- **Embedding generation** from protein sequences using state-of-the-art models
- **Distance heatmaps and 2D projections** (UMAP, t-SNE, PCA, PaCMAP) for visualization
- **Clustering evaluation** with multiple algorithms and comprehensive quality metrics
- **Interactive visualization** with dynamic filtering and upload features
- **Statistical analysis** for comparing embedding methods with significance testing
- Multiple input formats support (pickle, npz, hdf5, FASTA, TSV)
- Flexible output formats and customization options

## Requirements
- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scikit-learn
- umap-learn, colorcet, pacmap
- hdbscan (for HDBSCAN clustering)
- h5py (optional, for HDF5 input)
- dash, plotly (for interactive visualization)

## Supported Methods

### Clustering Algorithms
- **K-means**: Efficient partitional clustering with initialization options
- **Hierarchical**: Agglomerative clustering with multiple linkage criteria (ward, complete, average, single) and distance metrics
- **DBSCAN**: Density-based clustering for discovering clusters of varying shapes (auto-parameter search for eps/min_samples when not specified)
- **HDBSCAN**: Hierarchical density-based clustering with noise detection

### Dimensionality Reduction Methods
- **UMAP**: Uniform Manifold Approximation and Projection
- **t-SNE**: t-distributed Stochastic Neighbor Embedding
- **PCA**: Principal Component Analysis
- **PaCMAP**: Pairwise Controlled Manifold Approximation Projection ([GitHub](https://github.com/YingfanWang/PaCMAP))

### Evaluation Metrics
- **External validation**: Adjusted Rand Score, Normalized Mutual Information, Homogeneity, Completeness, V-Measure
- **Internal validation**: Silhouette Score, Calinski-Harabasz Score, Davies-Bouldin Score

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/cabd-bioinfo/rasembedd.git
cd rasembedd

# Install core dependencies
pip install -r requirements.txt

# Install clustering-specific dependencies
pip install -r requirements_clustering.txt

# Install interactive visualization dependencies (optional)
pip install -r requirements_interactive.txt
```

### Development Setup
```bash
# Run the development setup script (installs dependencies and runs tests)
bash setup_dev.sh
```
## Example Plots
Example plots are available in the [`exampleplots/`](exampleplots/) directory:

[![All heatmap cosine Family.name prost_t5 embeddings](exampleplots/all_heatmap_cosine_Family.name_prost_t5_embeddings.png)](exampleplots/all_heatmap_cosine_Family.name_prost_t5_embeddings.pdf)
[![All PCA projection Family.name prost_t5 embeddings](exampleplots/all_PCA_projection_Family.name_prost_t5_embeddings.png)](exampleplots/all_PCA_projection_Family.name_prost_t5_embeddings.pdf)
[![All TSNE projection Family.name prost_t5 embeddings](exampleplots/all_TSNE_projection_Family.name_prost_t5_embeddings.png)](exampleplots/all_TSNE_projection_Family.name_prost_t5_embeddings.pdf)
[![PaCMAP projection Family.name ankh_s2s embeddings](exampleplots/PaCMAP_projection_Family.name_ankh_s2s_embeddings.png)](exampleplots/PaCMAP_projection_Family.name_ankh_s2s_embeddings.pdf)

## Usage
For detailed usage instructions, see the individual script documentation:
- [Embedding Generation](README_generate_embeddings.md) - Generate embeddings from protein sequences
- [Visualization](README_generate_visualizations.md) - Create heatmaps and 2D projections
- [Clustering Evaluation](README_clustering_evaluation.md) - Evaluate clustering performance and compare methods
- [Interactive Visualization](README_interactive_visualizations.md) - Web-based interactive exploration
- [Model Development](README_build_model_module.md) - Add custom embedding models

## Interactive Visualization
A new interactive Dash app is available for exploring protein embeddings and metadata with dynamic filtering and upload features.

See the dedicated documentation here: [Interactive Visualization README](README_interactive_visualizations.md)

### Screenshot
![Interactive Visualization Screenshot](exampleplots/interactive.png)

---

# Complete Analysis Pipeline

This section describes the step-by-step pipeline for generating, visualizing, and evaluating protein embeddings.

## Pipeline Steps

### 1) Prepare Your Input Data
- **Format:** You can use either a TSV (tab-separated values) file or a FASTA file containing your protein sequences.
- **TSV:** Should include at least a unique identifier column (e.g., `uniprot_id`) and a sequence column (e.g., `sequence`).
- **FASTA:** Standard FASTA format with headers and sequences.

**Example:**
- Example TSV: [`data/RAS.updated.Info.Table.V7.tsv`](data/RAS.updated.Info.Table.V7.tsv)
- Example FASTA: [`data/RAS.fasta`](data/RAS.fasta)

```bash
# Example TSV
uniprot_id	sequence
P12345	MKTAYIAKQRQISFVKSHFSRQ...
...

# Example FASTA
>P12345
MKTAYIAKQRQISFVKSHFSRQ...
...
```

---

### 2) Generate Embeddings
- Use the [`generate_embeddings.py`](README_generate_embeddings.md) script to generate embeddings for your sequences using a selected model.
- If your desired model is not available, you can [build your own model module](README_build_model_module.md) and place it in the `models/` directory.

**Example Command:**
```bash
python generate_embeddings.py --input data/RAS.fasta --input_type fasta --model_type prot_t5 --output embeddings.pkl
```

- See the [embedding script documentation](README_generate_embeddings.md) for more details and options.
- See the [model module guide](README_build_model_module.md) to add custom models.
- Example output: [`embeddings/prost_t5_embeddings.pkl`](embeddings/prost_t5_embeddings.pkl)

---

### 3) Generate Plots and Visualizations
- Use the [`generate_visualizations.py`](README_generate_visualizations.md) script to create heatmaps and 2D projections (UMAP, t-SNE, PCA, PaCMAP) from your embeddings and metadata.

**Example Command:**
```bash
python generate_visualizations.py data/RAS.updated.Info.Table.V7.tsv -e embeddings.pkl
```

- See the [visualization script documentation](README_generate_visualizations.md) for all options and examples.

---

### 4) Evaluate Clustering Performance
- Use the [`clustering_evaluation.py`](README_clustering_evaluation.md) script to evaluate the quality of your protein embeddings by performing clustering analysis and comparing results against known protein classifications.

**Example Command:**
```bash
python clustering_evaluation.py embeddings.pkl data/metadata.tsv --methods kmeans hierarchical dbscan hdbscan
```

**Key Features:**
- **Multiple clustering algorithms**: K-means, Hierarchical (with linkage options), DBSCAN, HDBSCAN
- **Automatic optimization**: Finds optimal number of clusters using multiple criteria
- **Comprehensive evaluation**: 7+ clustering quality metrics including Adjusted Rand Score, V-Measure, Silhouette Score
- **Statistical analysis**: Subsampling with significance testing to compare embedding methods
- **Normalization options**: Standard, L2, PCA whitening, ZCA whitening, Pipeline, or None
- **Rich visualizations**: Cluster optimization plots, confusion matrices, significance heatmaps

- See the [clustering evaluation documentation](README_clustering_evaluation.md) for detailed usage and options.

---
- Example output files (click images for PDF):


#### Heatmap
[![Heatmap](exampleplots/all_heatmap_cosine_Family.name_prost_t5_embeddings.png)](exampleplots/all_heatmap_cosine_Family.name_prost_t5_embeddings.pdf)

#### PCA
[![PCA](exampleplots/all_PCA_projection_Family.name_prost_t5_embeddings.png)](exampleplots/all_PCA_projection_Family.name_prost_t5_embeddings.pdf)

#### t-SNE
[![t-SNE](exampleplots/all_TSNE_projection_Family.name_prost_t5_embeddings.png)](exampleplots/all_TSNE_projection_Family.name_prost_t5_embeddings.pdf)

#### PaCMAP
[![PaCMAP](exampleplots/PaCMAP_projection_Family.name_ankh_s2s_embeddings.png)](exampleplots/PaCMAP_projection_Family.name_ankh_s2s_embeddings.pdf)

---

## Summary
1. **Prepare your input**: TSV or FASTA file with protein sequences ([example TSV](data/RAS.updated.Info.Table.V7.tsv), [example FASTA](data/RAS.fasta)).
2. **Generate embeddings**: Use or build a model module, then run the embedding script ([example output](embeddings/prost_t5_embeddings.pkl)).
3. **Visualize**: Use the visualization script to generate publication-quality plots (see PNG previews above).
4. **Evaluate clustering**: Use the clustering evaluation script to assess embedding quality through clustering analysis and statistical comparison.

For more details, see the linked documentation files for each step.
