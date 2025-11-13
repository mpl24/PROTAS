# TCGA-PRAD Analysis

Differential gene expression analysis comparing high vs low reactive stroma (RS) in TCGA-PRAD cohort.

## Overview
Uses PROTAS slide-level features to stratify TCGA-PRAD samples, then performs DESeq2 analysis to identify RS-associated genes and pathways.


### 1. Download TCGA-PRAD Data (~15 minutes)

Create `data_download.R`:
```r
library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)

project <- 'TCGA-PRAD'

query <- GDCquery(
  project = project,
  data.category = 'Transcriptome Profiling',
  data.type = 'Gene Expression Quantification',
  workflow.type = 'STAR - Counts',
  sample.type = c('Primary Tumor', 'Solid Tissue Normal')
)

GDCdownload(query)
se <- GDCprepare(query)

counts <- assay(se)
rownames(counts) <- sub("\\.\\d+$", "", rownames(counts))
counts_df <- data.frame(gene_id = rownames(counts), counts, check.names = FALSE)
write.csv(counts_df, 'TCGA-PRAD_expression_matrix.csv', row.names = FALSE)

cd <- as.data.frame(colData(se))
meta <- cd %>%
  transmute(
    sample_id = barcode,
    patient = substr(barcode, 1, 12),
    sample = substr(barcode, 1, 16),
    sample_type = sample_type 
  )
write.csv(meta, 'TCGA-PRAD_metadata.csv', row.names = FALSE)
```
Run:
```bash
Rscript data_download.R
```

### 2. Preprocess with Slide-Level Features
```bash
python set_up_tcga_prad_data.py \
    --save_root ./preprocessed_data/ \
    --tcga_prad_metadata TCGA-PRAD_metadata.csv \
    --tcga_prad_clinical_matrix Xena_Clinical_Matrix.tsv \
    --slide_level_rs_features tcga_prad_slide_level_rs_features.csv \
    --raw_expression_data TCGA-PRAD_expression_matrix.csv
```
**Required files:**
- `tcga_prad_slide_level_rs_features.csv` - Provided in repository
- `Xena_Clinical_Matrix.tsv` - Download from [UCSC Xena](https://xenabrowser.net/datapages/?cohort=TCGA%20Prostate%20Cancer%20(PRAD))

**Output:**
- `preprocessed_data/expression_matrix_mean_prob_rs.csv` - Filtered expression matrix
- `preprocessed_data/metadata_mean_prob_rs.csv` - Metadata with RS groups


## Dependencies

### Python
```bash
pip install pandas numpy
```

### R
```r
install.packages(c("dplyr", "tibble", "readr", "ggplot2"))

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c(
  "TCGAbiolinks", 
  "DESeq2", 
  "clusterProfiler", 
  "EnhancedVolcano", 
  "org.Hs.eg.db", 
  "biomaRt",
  "enrichplot",
  "GOplot"
))
```

## Analysis Parameters

- **RS grouping**: Top 75th percentile vs rest
- **DEG cutoffs**: padj < 0.05, |log2FC| > 1
- **Gene filtering**: Protein-coding genes only
- **GO enrichment**: Biological Process (BP)
- **GSEA**: 10,000 permutations, minGSSize=10, maxGSSize=500

## Notes

- Download time: ~15 minutes (depends on connection)
- TCGA-PRAD has ~400 samples
- Grade group mapping: GG 1-2 → low, GG 3-5 → high
- `tcga_prad_slide_level_rs_features.csv` must be in repository
