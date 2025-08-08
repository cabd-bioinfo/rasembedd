# Clustering Evaluation Script

A comprehensive tool for evaluating protein clustering based on embeddings. This script analyzes the quality of protein embeddings by performing clustering analysis and comparing results against known protein classifications.

## Overview

The clustering evaluation script performs the following key tasks:

1. Load protein embeddings from pickle files and metadata from TSV/CSV files
2. Perform clustering using multiple algorithms (K-means, Hierarchical, Spectral, DBSCAN, HDBSCAN)
3. Optimize cluster numbers automatically or use user-specified values
4. Evaluate clustering quality using multiple metrics against ground truth labels
5. Generate comprehensive visualizations and statistical reports
6. Support subsampling analysis for robustness testing

## Features

### Clustering Methods
- K-means: Efficient partitional clustering with initialization options
- Hierarchical: Agglomerative clustering with multiple linkage criteria (ward, complete, average, single) and distance metrics (euclidean, manhattan, cosine, etc.)
- Spectral: Graph-based clustering using eigenvectors of an affinity matrix (rbf or nearest_neighbors); participates in K optimization
- DBSCAN: Density-based clustering for discovering clusters of varying shapes; automatically searches eps/min_samples if not provided
- HDBSCAN: Hierarchical density-based clustering with noise detection; evaluated once and reports the observed number of clusters (excluding noise)

### Evaluation Metrics
- Adjusted Rand Score: Similarity measure corrected for chance
- Normalized Mutual Information: Information-theoretic measure
- Homogeneity & Completeness: Cluster purity measures
- V-Measure: Harmonic mean of homogeneity and completeness
- Silhouette Score: Internal cluster quality measure
- Calinski-Harabasz Score: Variance ratio criterion
- Davies-Bouldin Score: Average similarity measure

### Visualizations
- Cluster optimization plots: Metrics vs number of clusters
- Truth tables: Confusion matrices showing clustering accuracy
- Significance heatmaps: Statistical comparison between embedding methods
- Cluster assignment files: Detailed results for each protein

### Advanced Features
- Automatic cluster optimization: Finds optimal number of clusters (K-means/Hierarchical)
    - Also applies to Spectral Clustering
- Subsampling analysis: Tests robustness across multiple random samples
- Statistical testing: Compares different embedding methods with significance tests
- Stratified subsampling: Maintains class proportions in samples
- Multiple normalization methods: Standard, L2, PCA whitening, ZCA whitening, Pipeline, or None

### Normalization Methods
- Standard (Z-score): Centers features to mean=0, scales to std=1
- L2: Normalizes each sample to unit norm (magnitude=1) - default
- PCA Whitening: Decorrelates features and scales to unit variance
- ZCA Whitening: Decorrelates while preserving original feature relationships
- Pipeline: Configurable 3-step pipeline: center/scale → PCA (n components or variance) → optional L2
- None: No normalization applied

## Requirements

Install the required dependencies:

```bash
pip install -r requirements_clustering.txt
```

### Core Dependencies
- numpy, pandas, matplotlib, seaborn
- scikit-learn: Clustering algorithms and metrics
- hdbscan: HDBSCAN clustering algorithm
- umap-learn: UMAP dimensionality reduction
- statsmodels: Statistical testing
- joblib: Parallel processing

## Usage

### Basic Usage

```bash
python clustering_evaluation.py embeddings1.pkl embeddings2.pkl metadata.tsv
```

### Advanced Examples

Specify clustering methods and parameters:
```bash
python clustering_evaluation.py \
    embeddings/*.pkl \
    metadata/protein_metadata.tsv \
    --methods kmeans hierarchical spectral dbscan hdbscan \
    --max-clusters 20 \
    --normalization-method standard \
    --output-dir results/
```

Use different normalization methods:
```bash
# Standard normalization (z-score)
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method standard

# L2 normalization (unit norm)
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method l2

# PCA whitening
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method pca

# No normalization
python clustering_evaluation.py embeddings.pkl metadata.tsv --normalization-method none

# Configurable pipeline (center/scale → PCA by variance → L2)
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --normalization-method pipeline \
    --norm-center --norm-scale \
    --norm-pca-components 0.95 \
    --norm-l2
```

Customize clustering algorithm parameters:
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

# DBSCAN with automatic parameter search (omit both flags)
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods dbscan

# K-means with specific initialization and iterations
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods kmeans \
    --kmeans-init random \
    --kmeans-max-iter 500

# Spectral clustering with nearest-neighbors affinity
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods spectral \
    --n-clusters 3 \
    --spectral-affinity nearest_neighbors \
    --spectral-n-neighbors 10 \
    --spectral-assign-labels kmeans
```

Run subsampling analysis:
```bash
python clustering_evaluation.py \
    embeddings1.pkl embeddings2.pkl \
    metadata.tsv \
    --subsample 100 \
    --subsample-fraction 0.8 \
    --stratified-subsample
```

Custom metadata columns:
```bash
python clustering_evaluation.py \
    embeddings.pkl \
    metadata.tsv \
    --id-column "protein_id" \
    --label-column "family_classification"
```

Combined subsampling + pipeline example:
```bash
python clustering_evaluation.py embeddings.pkl metadata.tsv \
    --methods kmeans dbscan \
    --normalization-method pipeline \
    --norm-center --norm-scale --norm-pca-components 0.9 --norm-l2 \
    --subsample 50 --subsample-fraction 0.7 --stratified-subsample
```

## Command Line Arguments

### Required Arguments
- embedding_files: One or more paths to embedding pickle files
- metadata_file: Path to metadata file (TSV or CSV format)

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --output-dir, -o | clustering_results | Output directory for results |
| --id-column | uniprot_id | Column name for protein IDs in metadata |
| --label-column | Family.name | Column name for true labels in metadata |
| --methods | kmeans hierarchical | Clustering methods: kmeans, hierarchical, dbscan, hdbscan |
| --methods | kmeans hierarchical | Clustering methods: kmeans, hierarchical, spectral, dbscan, hdbscan |
| --n-clusters | Auto-optimize | Number of clusters (fixed) |
| --max-clusters | 15 | Maximum clusters for optimization |
| --normalization-method | l2 | Normalization method: standard, l2, pca, zca, pipeline, none |
| --subsample | 0 | Number of subsampling runs |
| --subsample-fraction | 0.8 | Fraction of proteins per subsample |
| --stratified-subsample | False | Use stratified subsampling |

Note: When using normalization-method pipeline, configure steps with the flags below.

### Clustering Algorithm Parameters

#### K-means Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| --kmeans-init | k-means++ | Initialization method: k-means++, random |
| --kmeans-max-iter | 300 | Maximum number of iterations |

#### Hierarchical Clustering Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| --hierarchical-linkage | complete | Linkage criterion: ward, complete, average, single |
| --hierarchical-metric | euclidean | Distance metric: euclidean, manhattan, cosine, l1, l2 |

Note: Ward linkage only supports euclidean metric. For other metrics like manhattan, use complete, average, or single linkage.

#### DBSCAN Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| --dbscan-eps | 0.5 | Maximum distance between samples in a neighborhood |
| --dbscan-min-samples | 5 | Minimum samples in a neighborhood for core point |

If either parameter is omitted, the script performs a small parameter search using k-distance quantiles to propose eps candidates and tries a set of min_samples values. A composite score (valid clusters, silhouette, ARI, and Davies-Bouldin penalty) selects the best combination. All-noise solutions are rejected.

#### HDBSCAN Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| --hdbscan-min-cluster-size | 5 | Minimum size of clusters |
| --hdbscan-min-samples | None | Minimum samples for core points (None uses min_cluster_size) |
| --hdbscan-cluster-selection-epsilon | 0.0 | Distance threshold for cluster selection |

HDBSCAN is evaluated once per configuration; the observed number of clusters (excluding noise points labeled as -1) is reported in the results.

#### Spectral Clustering Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| --spectral-affinity | rbf | Affinity function: rbf, nearest_neighbors, precomputed, precomputed_nearest_neighbors |
| --spectral-assign-labels | kmeans | Label assignment strategy: kmeans, discretize |
| --spectral-n-neighbors | 10 | Number of neighbors for nearest_neighbors affinity |
| --spectral-gamma | auto | RBF kernel coefficient (omit flag to use sklearn default) |

#### Normalization Pipeline Options (when --normalization-method pipeline)
| Argument | Description |
|----------|-------------|
| --norm-center / --norm-no-center | Enable/disable feature centering (mean=0) |
| --norm-scale / --norm-no-scale | Enable/disable feature scaling (std=1) |
| --norm-pca-components N | PCA dimensionality: integer N components, or a float in (0,1] for variance retained (e.g., 0.95) |
| --norm-l2 / --norm-no-l2 | Enable/disable L2 normalization of samples after PCA |

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
uniprot_id\tFamily.name\tspecies\tdescription
P12345\tkinase\thuman\tProtein kinase
Q67890\ttranscription_factor\tmouse\tTranscription factor
...
```

## Output Files

The script generates several output files in the specified directory:

### Result Files
- {embedding_name}_clustering_results.tsv: Detailed metrics for each clustering run (includes params_json)
- {embedding_name}_cluster_assignments.tsv: Cluster assignments for each protein
- embedding_clustering_summary.tsv: Summary comparison across all embeddings
- embedding_clustering_parameters.tsv: Flat audit log of the exact parameters used per run (including DBSCAN/HDBSCAN and normalization settings)

### Visualizations
- cluster_optimization_{embedding}_{method}.pdf: Optimization plots showing metrics vs cluster count
- truth_table_{embedding}_{method}.pdf: Confusion matrices comparing clusters to true labels
- significance_heatmap_{comparison}.pdf: Statistical comparison heatmaps

### Statistical Analysis
- statistical_tests_{comparison}.tsv: P-values and effect sizes for embedding comparisons
- Subsampling results with confidence intervals and statistical tests

## Example Workflow

1. Prepare your data
2. Run basic clustering evaluation
3. Compare multiple embedding methods
4. Analyze results

## Interpretation Guide

### Key Metrics to Watch
- Adjusted Rand Score (higher is better, 0-1)
- Silhouette Score (higher is better, -1 to 1)
- V-Measure (higher is better, 0-1)

### Cluster Optimization
- Look for elbow points in optimization curves
- Consider multiple metrics when choosing optimal cluster count
- Balance internal metrics (silhouette) with external validation (ARI)

### Statistical Significance
- P-values < 0.05 indicate significant differences
- Effect sizes help assess practical significance
- Holm-Bonferroni correction for multiple testing

## Troubleshooting

Common issues and tips:
- Missing protein IDs: ensure exact matches between embeddings and metadata
- Memory issues: use subsampling or different normalization
- Poor clustering: try different methods, normalization, or parameter ranges
- Parameter compatibility: ward linkage requires euclidean; DBSCAN defaults may yield many noise points
- Visualization errors: check matplotlib backend and output directory permissions

## Technical Details

### Algorithm Implementation
- K-means: scikit-learn KMeans with configurable init and iterations
- Hierarchical: Agglomerative clustering with configurable linkage and metrics
- Spectral: scikit-learn SpectralClustering with configurable affinity and label assignment; optimized over K like K-means/Hierarchical
- DBSCAN: Density-based clustering with eps/min_samples; auto-search when not provided
- HDBSCAN: Hierarchical density-based clustering with noise detection

### Density-based Optimization
- DBSCAN skips k-optimization and instead searches over eps (via k-distance quantiles) and min_samples when not provided. A composite score selects the best model, preferring valid multi-cluster solutions and higher silhouette/ARI, with a penalty for Davies-Bouldin.
- HDBSCAN is run once for the provided parameters; the observed cluster count excludes noise. If all points are noise, the reported cluster count is 0.

### Statistical Testing
- Paired t-tests, Wilcoxon signed-rank tests, Holm-Bonferroni correction

### Memory Management
- Streaming I/O, efficient numpy ops, optional garbage collection

## Recent Updates

### Version Improvements
- HDBSCAN Support: Added hierarchical density-based clustering with noise detection
- Enhanced Hierarchical Clustering: Multiple linkage criteria and distance metrics
- Parameter Validation: Improved error handling and compatibility checking
- Better Defaults: Default hierarchical linkage from ward to complete
- Warning Management: Suppressed deprecation warnings from external libraries

### New Features
- DBSCAN automatic parameter search (eps/min_samples) with composite scoring and noise handling
- Correct handling of density-based methods in optimization (skip k-optimization; report observed clusters)
- Normalization pipeline with CLI flags for center/scale, PCA by components or variance, and optional L2
- Results now include a params_json column and a new embedding_clustering_parameters.tsv audit file

### What's New (2025-08-08)
- DBSCAN now performs an automatic parameter search when eps/min_samples are not provided.
- HDBSCAN is evaluated once per configuration; reported cluster counts exclude noise.
- Added a configurable normalization pipeline (center/scale → PCA → L2) with CLI flags.
- Results include params_json and an audit file embedding_clustering_parameters.tsv.
 - Added Spectral Clustering support (rbf or nearest_neighbors affinity), participates in K optimization and has dedicated CLI flags.

## Citation

If you use this clustering evaluation script in your research, please cite the appropriate embedding methods and this analysis framework.

## License

This script is part of the rasembedd project. See the main project README for license information.
