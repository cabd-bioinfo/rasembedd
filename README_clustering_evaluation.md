# Clustering Evaluation Script

A comprehensive tool for evaluating protein clustering based on embeddings. This script analyzes the quality of protein embeddings by performing clustering analysis and comparing results against known protein classifications.

## Overview

The clustering evaluation script performs the following key tasks:

1. **Load protein embeddings** from pickle files and metadata from TSV/CSV files
2. **Perform clustering** using multiple algorithms (K-means, Hierarchical, DBSCAN, HDBSCAN)
3. **Optimize cluster numbers** automatically or use user-specified values
4. **Evaluate clustering quality** using multiple metrics against ground truth labels
5. **Generate comprehensive visualizations** and statistical reports
6. **Support subsampling analysis** for robustness testing

## Features

### Clustering Methods
- **K-means**: Efficient partitional clustering with initialization options
- **Hierarchical**: Agglomerative clustering with multiple linkage criteria (ward, complete, average, single) and distance metrics (euclidean, manhattan, cosine, etc.)
- **DBSCAN**: Density-based clustering for discovering clusters of varying shapes
- **HDBSCAN**: Hierarchical density-based clustering with noise detection and automatic parameter selection

### Evaluation Metrics
- **Adjusted Rand Score**: Similarity measure corrected for chance
- **Normalized Mutual Information**: Information-theoretic measure
- **Homogeneity & Completeness**: Cluster purity measures
- **V-Measure**: Harmonic mean of homogeneity and completeness
- **Silhouette Score**: Internal cluster quality measure
- **Calinski-Harabasz Score**: Variance ratio criterion
- **Davies-Bouldin Score**: Average similarity measure

### Visualizations
- **Cluster optimization plots**: Metrics vs number of clusters
- **Truth tables**: Confusion matrices showing clustering accuracy
- **Significance heatmaps**: Statistical comparison between embedding methods
- **Cluster assignment files**: Detailed results for each protein

### Advanced Features
- **Automatic cluster optimization**: Finds optimal number of clusters
- **Subsampling analysis**: Tests robustness across multiple random samples
- **Statistical testing**: Compares different embedding methods with significance tests
- **Stratified subsampling**: Maintains class proportions in samples
- **Multiple normalization methods**: Standard, L2, PCA whitening, ZCA whitening, or none

### Normalization Methods
- **Standard (Z-score)**: Centers features to mean=0, scales to std=1
- **L2**: Normalizes each sample to unit norm (magnitude=1) - **default**
- **PCA Whitening**: Decorrelates features and scales to unit variance
- **ZCA Whitening**: Decorrelates while preserving original feature relationships
- **None**: No normalization applied

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_clustering.txt
```

### Core Dependencies
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`: Clustering algorithms and metrics
- `hdbscan`: HDBSCAN clustering algorithm
- `umap-learn`: UMAP dimensionality reduction
- `statsmodels`: Statistical testing
- `joblib`: Parallel processing

### Optional Dependencies
- `colorcet`: Enhanced color palettes for visualizations

## Usage

### Basic Usage

```bash
python clustering_evaluation.py embeddings1.pkl embeddings2.pkl metadata.tsv
```

### Advanced Examples

**Specify clustering methods and parameters:**
```bash
python clustering_evaluation.py \
    embeddings/*.pkl \
    metadata/protein_metadata.tsv \
    --methods kmeans hierarchical dbscan hdbscan \
    --max-clusters 20 \
    --normalization-method standard \
    --output-dir results/
```

**Use different normalization methods:**
```bash
# Standard normalization (z-score)
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method standard

# L2 normalization (unit norm)
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method l2

# PCA whitening
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method pca

# No normalization
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method none
```

**Customize clustering algorithm parameters:**
```bash
# Hierarchical clustering with manhattan distance and complete linkage
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods hierarchical \
    --hierarchical-metric manhattan \
    --hierarchical-linkage complete

# HDBSCAN with custom parameters
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods hdbscan \
    --hdbscan-min-cluster-size 10 \
    --hdbscan-min-samples 5 \
    --hdbscan-cluster-selection-epsilon 0.1

# DBSCAN with custom eps and min_samples
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods dbscan \
    --dbscan-eps 0.3 \
    --dbscan-min-samples 3

# K-means with specific initialization and iterations
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods kmeans \
    --kmeans-init random \
    --kmeans-max-iter 500
```

**Run subsampling analysis:**
```bash
python clustering_evaluation.py \
    embeddings1.pkl embeddings2.pkl \
    metadata.tsv \
    --subsample 100 \
    --subsample-fraction 0.8 \
    --stratified-subsample
```

**Custom metadata columns:**
```bash
python clustering_evaluation.py \
    embeddings.pkl \
    metadata.tsv \
    --id-column "protein_id" \
    --label-column "family_classification"
```

## Command Line Arguments

### Required Arguments
- `embedding_files`: One or more paths to embedding pickle files
- `metadata_file`: Path to metadata file (TSV or CSV format)

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir`, `-o` | `clustering_results` | Output directory for results |
| `--id-column` | `uniprot_id` | Column name for protein IDs in metadata |
| `--label-column` | `Family.name` | Column name for true labels in metadata |
| `--methods` | `kmeans hierarchical` | Clustering methods: kmeans, hierarchical, dbscan, hdbscan |
| `--n-clusters` | Auto-optimize | Number of clusters (fixed) |
| `--max-clusters` | `15` | Maximum clusters for optimization |
| `--normalization-method` | `l2` | Normalization method: standard, l2, pca, zca, none |
| `--subsample` | `0` | Number of subsampling runs |
| `--subsample-fraction` | `0.8` | Fraction of proteins per subsample |
| `--stratified-subsample` | `False` | Use stratified subsampling |

### Clustering Algorithm Parameters

#### K-means Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--kmeans-init` | `k-means++` | Initialization method: k-means++, random |
| `--kmeans-max-iter` | `300` | Maximum number of iterations |

#### Hierarchical Clustering Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--hierarchical-linkage` | `complete` | Linkage criterion: ward, complete, average, single |
| `--hierarchical-metric` | `euclidean` | Distance metric: euclidean, manhattan, cosine, l1, l2 |

**Note:** Ward linkage only supports euclidean metric. For other metrics like manhattan, use complete, average, or single linkage.

#### DBSCAN Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--dbscan-eps` | `0.5` | Maximum distance between samples in a neighborhood |
| `--dbscan-min-samples` | `5` | Minimum samples in a neighborhood for core point |

#### HDBSCAN Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--hdbscan-min-cluster-size` | `5` | Minimum size of clusters |
| `--hdbscan-min-samples` | `None` | Minimum samples for core points (None uses min_cluster_size) |
| `--hdbscan-cluster-selection-epsilon` | `0.0` | Distance threshold for cluster selection |

## Input File Formats

### Embedding Files (.pkl)
Python pickle files containing dictionaries mapping protein IDs to numpy arrays:

```python
{
    "protein_1": np.array([0.1, 0.2, 0.3, ...]),
    "protein_2": np.array([0.4, 0.5, 0.6, ...]),
    ...
}
```

### Metadata Files (.tsv/.csv)
Tab-separated or comma-separated files with columns for protein IDs and classifications:

```
uniprot_id	Family.name	species	description
P12345	kinase	human	Protein kinase
Q67890	transcription_factor	mouse	Transcription factor
...
```

## Output Files

The script generates several output files in the specified directory:

### Result Files
- `{embedding_name}_clustering_results.tsv`: Detailed metrics for each clustering run
- `{embedding_name}_cluster_assignments.tsv`: Cluster assignments for each protein
- `embedding_clustering_summary.tsv`: Summary comparison across all embeddings

### Visualizations
- `cluster_optimization_{embedding}_{method}.pdf`: Optimization plots showing metrics vs cluster count
- `truth_table_{embedding}_{method}.pdf`: Confusion matrices comparing clusters to true labels
- `significance_heatmap_{comparison}.pdf`: Statistical comparison heatmaps

### Statistical Analysis
- `statistical_tests_{comparison}.tsv`: P-values and effect sizes for embedding comparisons
- Subsampling results with confidence intervals and statistical tests

## Example Workflow

1. **Prepare your data:**
   ```bash
   # Ensure embeddings are saved as pickle files
   # Ensure metadata contains protein IDs and classifications
   ```

2. **Run basic clustering evaluation:**
   ```bash
   python clustering_evaluation.py embeddings/prost_t5.pkl metadata/families.tsv
   ```

3. **Compare multiple embedding methods:**
   ```bash
   python clustering_evaluation.py \
       embeddings/prost_t5.pkl \
       embeddings/ankh.pkl \
       embeddings/esm.pkl \
       metadata/families.tsv \
       --output-dir comparison_results/
   ```

4. **Analyze results:**
   - Check `embedding_clustering_summary.tsv` for overall performance
   - View PDF plots for detailed cluster optimization curves
   - Examine truth tables to understand clustering accuracy
   - Review statistical tests for significant differences

## Interpretation Guide

### Key Metrics to Watch
- **Adjusted Rand Score** (higher is better, 0-1): Measures clustering accuracy against true labels
- **Silhouette Score** (higher is better, -1 to 1): Internal cluster quality
- **V-Measure** (higher is better, 0-1): Balance of homogeneity and completeness

### Cluster Optimization
- Look for "elbow" points in optimization curves
- Consider multiple metrics when choosing optimal cluster count
- Balance internal metrics (silhouette) with external validation (ARI)

### Statistical Significance
- P-values < 0.05 indicate significant differences between embedding methods
- Effect sizes help assess practical significance beyond statistical significance
- Multiple testing correction (Holm method) controls false discovery rate

## Troubleshooting

### Common Issues

**Missing protein IDs:**
- Ensure protein IDs match exactly between embeddings and metadata
- Check for case sensitivity or formatting differences

**Memory issues with large datasets:**
- Use subsampling: `--subsample 50 --subsample-fraction 0.5`
- Try different normalization methods: `--normalization-method standard` or `--normalization-method pca`

**Poor clustering results:**
- Try different clustering methods: `--methods kmeans hierarchical dbscan hdbscan`
- Experiment with normalization: `--normalization-method standard` or `--normalization-method none`
- Adjust cluster count range: `--max-clusters 25`
- For hierarchical clustering with non-euclidean metrics, use `--hierarchical-linkage complete`
- For HDBSCAN, try adjusting `--hdbscan-min-cluster-size` based on expected cluster sizes
- Check if true labels are meaningful for clustering

**Parameter compatibility issues:**
- Ward linkage only works with euclidean metric. Use `--hierarchical-linkage complete` with `--hierarchical-metric manhattan`
- HDBSCAN parameters: `min_samples` defaults to `min_cluster_size` if not specified
- DBSCAN may produce many noise points (-1 labels) with default parameters - adjust `eps` and `min_samples`

**Visualization errors:**
- Ensure matplotlib backend is properly configured
- Check write permissions in output directory

### Performance Tips
- Use subsampling for large datasets (>10,000 proteins)
- Choose appropriate normalization: `l2` (default) for most embeddings, `standard` for mixed scales, `none` for pre-normalized data
- Use stratified subsampling to maintain class balance
- Consider parallel processing for multiple embedding comparisons

## Technical Details

### Algorithm Implementation
- **K-means**: Uses scikit-learn's KMeans with configurable initialization (k-means++, random) and iteration limits
- **Hierarchical**: Agglomerative clustering with configurable linkage (ward, complete, average, single) and distance metrics (euclidean, manhattan, cosine, etc.)
- **DBSCAN**: Density-based clustering with configurable eps and min_samples parameters
- **HDBSCAN**: Hierarchical density-based clustering with automatic parameter selection and noise detection

### Parameter Validation
- Ward linkage automatically validates euclidean metric requirement
- HDBSCAN parameters are validated for logical consistency
- All clustering methods include proper error handling for edge cases

### Statistical Testing
- Paired t-tests for normally distributed metrics
- Wilcoxon signed-rank tests for non-parametric comparisons
- Holm-Bonferroni correction for multiple testing

### Memory Management
- Streaming data processing for large embedding files
- Efficient numpy operations for metric calculations
- Optional garbage collection for memory-constrained environments

## Recent Updates

### Version Improvements
- **HDBSCAN Support**: Added hierarchical density-based clustering with noise detection
- **Enhanced Hierarchical Clustering**: Support for multiple linkage criteria and distance metrics
- **Parameter Validation**: Improved error handling and parameter compatibility checking
- **Better Defaults**: Changed default hierarchical linkage from 'ward' to 'complete' for better metric compatibility
- **Warning Management**: Suppressed deprecation warnings from external libraries

### New Features
- Complete parameter customization for all clustering algorithms
- Automatic validation of incompatible parameter combinations
- Enhanced documentation and error messages
- Support for manhattan, cosine, and other distance metrics in hierarchical clustering

## Citation

If you use this clustering evaluation script in your research, please cite the appropriate embedding methods and this analysis framework.

## License

This script is part of the rasembedd project. See the main project README for license information.
