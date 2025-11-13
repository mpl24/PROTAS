# Statistical Analysis

Statistical methods for analyzing PROTAS features and survival outcomes.

## Overview

Tools for Cox proportional hazards regression, feature normalization, model comparison, and association testing.

## Contents

### 1. Survival Analysis (`cox_ph.py`)

Cox regression with bootstrapped confidence intervals and model comparison.

**Usage:**
```bash
python cox_ph_model/cox_ph.py --dataset merged_features.csv
```

**Models tested:**
- `RS_only` - Reactive stroma features only
- `RS_plus_tumor` - RS + tumor morphology
- `RS_plus_capra` - RS + CAPRA score
- `Capra_only` - Clinical risk score only
- `Tumor_only` - Tumor features only
- `All` - Complete feature set

**Key functions:**
- `enhanced_fit_cph()` - Fit Cox model with bootstrapping
- `bootstrap_individual_model()` - Bootstrap C-index and time-dependent AUC
- `bootstrap_between_models()` - Pairwise model comparison
- `get_cross_model_results()` - Compare all model pairs

**Output:**
- Hazard ratios (HR) with 95% CI
- C-index with 95% CI
- Time-dependent AUC at 12, 24, 60 months
- P-values (FDR-adjusted)

### 2. Feature Normalization (`./cox_ph_model/preprocessing.py`)

Normalize features for survival analysis.

**Normalization methods:**
- `YJ` - Yeo-Johnson transformation (for skewed data)
- `log` - Log1p transformation
- `standard` - Z-score standardization

**Usage:**
```bash
python ./cox_ph_model/preprocessing.py \
    --dataset_root features.csv \
    --output_dir ./scaled_features/
```

**Features:**
- Applies appropriate transformation per feature
- Creates binned versions for categorical analysis
- Visualizes distributions before/after scaling


### 3. Group Comparisons (`wilcoxon_rank_sum_analysis.py`)

Mann-Whitney U tests for comparing feature distributions across grade groups.

**Usage:**
```bash
python wilcoxon_rank_sum_analysis.py --slide_level_features features.csv
```

**Tests:**
- Low grade (GG 1-2) vs High grade (GG 3-5)
- Bonferroni correction for multiple testing
- Reports adjusted p-values

### 4. Patch-Level GLMMs (R)

Generalized linear mixed models for patch-level features (nuclei, collagen).

**Usage:**
```r
# Edit script to set paths
Rscript glmm_nuclei.R
Rscript glmm_collagen.R
```

**Model:**
```r
glm(rs_label ~ feature + as.factor(slide_name),
    family = binomial(link = "logit"))
```

**Features:**
- Accounts for slide-level clustering
- Tests association with RS label
- Reports odds ratios with 95% CI
