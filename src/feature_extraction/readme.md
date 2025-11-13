# Feature Extraction
Extract slide-level features from PROTAS patch predictions for survival analysis.

## Overview
Takes per-patch RS predictions and extracts spatial features for downstream analysis.


## Quick Start

```bash
# Extract features
python ./slide_level_features/extract_per_slide_features.py \
    --dataframe_root /path/to/predictions/ \
    --clinical_data /path/to/clinical.csv \
    --save_root /path/to/output/

# Extract cancer features
python ./tumor_features/extract_tumor_features.py \
    --cancer_mask_root /path/to/cancer_masks/ \
    --tissue_mask_root /path/to/tissue_masks/ \
    --stroma_info_within_cancer_csv /path/to/stroma_info.csv \
    --save_root /path/to/output/
```


## Input
**Per-slide CSVs** in `dataframe_root/`:
- Columns: `x_loc`, `y_loc`, `mean_pred`, `confident`, `distance`
- One CSV per slide: `slide_name.csv`

## Output
basic_features - RS prevalence and distribution
hotspot_features - Spatial clustering patterns
distance_features - RS by distance from cancer
hotspot_graph_features - Network topology
cancer_features - Cancer morphology

## Key Features
Basic: mean_prob, rs_pos_percent, entropy_prob
Hotspot: num_hotspots, percent_hotspot, nni, center_of_mass_x/y
Distance: rs_edge_core_ratio, rs_com_distance, rs_rate_bin_Nmm
Graph: num_nodes, clustering_coeff, mean_betweenness
Cancer: percent_cancer_in_tissue, region_area, stroma_to_epithelial_ratio

## Dependencies
```bash
pip install numpy pandas scipy scikit-learn scikit-image networkx
```
