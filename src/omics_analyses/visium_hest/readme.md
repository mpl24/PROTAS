# Visium Spatial Transcriptomics Analysis

Spatial differential gene expression (DGE) analysis using PROTAS reactive stroma predictions on HEST-1K Visium data.

## Overview

Performs spatially-aware DGE analysis comparing high vs low RS regions within Visium slides using linear mixed models (LMMs) with spatial covariance kernels.

## Data Download

Download prostate Visium samples from HEST-1K:

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download HEST-1K dataset
huggingface-cli download MahmoodLab/hest --repo-type dataset --local-dir ./HEST-1K/
```

**Dataset**: [https://huggingface.co/datasets/MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest)

**Prostate samples used:**
- MEND151
- MEND162
- MEND60
- MEND62

## Quick Start

### 1. Preprocess Visium Data

```bash
python ./preprocessing/set_up_hest_data.py \
    --h5ad_root /path/to/HEST-1K/st/ \
    --stroma_RS_predictions ./preprocessing/stroma_rs_predictions.csv \
    --save_root ./preprocessed_visium/
```

**Input:**
- `.h5ad` files from HEST-1K
- `stroma_rs_predictions.csv` - Provided in repository (PROTAS predictions per spot)

**Output:**
- `counts_*.csv` - Normalized expression per sample
- `coords_*.csv` - Spatial coordinates per sample
- `filtered_metadata_*.csv` - Metadata with RS predictions per sample
- `expression_matrix.csv` - Combined expression across samples
- `metadata.csv` - Combined metadata

**Preprocessing steps:**
- Filter: min_counts=50, min_cells=5
- Normalize: target_sum=10,000
- Log-transform: log1p

### 2. Run Spatial DGE Analysis

```bash
python spatial_dge.py \
    --data_dir ./preprocessed_visium/sub_patches_min_counts_50_min_cells_5/ \
    --output_dir ./spatial_dge_results/
```

**Method:**
- Linear mixed model (LMM) with spatial covariance kernel
- Gaussian RBF kernel: `K(x,y) = exp(-||x-y||²/(2σ²))`
- Length scale optimization per sample
- FDR correction (Benjamini-Hochberg)


## Output

### Per-sample results:
- `{sample_id}_results_optimized_length.csv`
  - Columns: `gene`, `pvalue`, `padj`, `lml_null`, `lml_alt`, `scale`, `log_lr`, `optimal_length_scale`

### Combined results:
- `all_samples_combined_results_optimized_length.csv` - All samples merged
- `sample_summary.csv` - Summary statistics per sample

**Summary statistics:**
- `n_spots` - Number of Visium spots
- `n_genes_tested` - Genes tested
- `n_sig_05`, `n_sig_01` - Significant genes at FDR < 0.05/0.01
- `optimal_length_scale` - Optimized spatial kernel width (in pixels)


## Statistical Model

### LMM Formulation

```
Y = Xβ + Zu + ε

where:
  Y = gene expression (normalized, log-transformed)
  X = fixed effects (intercept, RS prediction)
  Z = random effects (spatial correlation)
  K = spatial covariance kernel
  u ~ N(0, K)
```

### Spatial Kernel

Radial basis function (RBF) with optimized length scale:

```python
K(x_i, x_j) = exp(-||x_i - x_j||² / (2 * σ²))
```

Length scales tested: [20, 40, 60, 80, 100, 120] pixels

Optimal σ selected by maximizing marginal likelihood on 20 random genes.

### Hypothesis Testing

- **Null model**: Gene expression ~ Intercept + Spatial effects
- **Alternative model**: Gene expression ~ Intercept + RS_prediction + Spatial effects
- **Test statistic**: Likelihood ratio test (LRT)
- **Multiple testing**: Benjamini-Hochberg FDR correction


## Dependencies

```bash
pip install numpy pandas scipy scikit-learn scanpy anndata h5py limix numba statsmodels
```

**Key packages:**
- `scanpy` - Visium data processing
- `limix` - LMM implementation
- `numba` - Fast kernel computation
- `statsmodels` - Multiple testing correction

## Notes

- Spots within cancer regions are excluded
- Only spots with confident RS predictions are used
- Genes filtered by min expression (min_counts=50, min_cells=5)
- Spatial kernel regularized with small diagonal term (1e-3)