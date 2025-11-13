# TCGA-PRAD Analysis

Differential gene expression analysis comparing high vs low reactive (RS) in TCGA-PRAD cohort.

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
Rscript download_tcga_prad.R
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

### 3. Run DESeq2 Analysis

```r
library(DESeq2)
library(biomaRt)

counts <- read.csv("./preprocessed_data/expression_matrix.csv", check.names = FALSE)
meta <- read.csv("./preprocessed_data/metadata.csv")

# Prepare data
ensembl_ids <- counts$gene_name
rownames(counts) <- counts$gene_name
counts$gene_name <- NULL
rownames(meta) <- meta$sample_id
meta$sample_id <- NULL
meta$X <- NULL

all(colnames(counts) == rownames(meta)) 

dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData = meta,
  design = ~ tas_group_percentile  
)

dds <- DESeq(dds)
res <- results(dds) 
write.csv(as.data.frame(res), "./preprocessed_data/deseq2_results.csv")

ensembl <- useMart("ensembl", dataset="hsapiens_gene_ensembl")
ensembl_to_gensymbol <- getBM(
  attributes = c("ensembl_gene_id", 'hgnc_symbol'), 
  filters = 'ensembl_gene_id', 
  values = ensembl_ids, 
  mart = ensembl
)
write.csv(as.data.frame(ensembl_to_gensymbol), 
          "./preprocessed_data/gene_symbols.csv")

gene_biotypes <- getBM(
  attributes = c("ensembl_gene_id", "hgnc_symbol", "gene_biotype"),
  filters = "ensembl_gene_id",
  values = ensembl_ids,
  mart = ensembl
)

coding_genes <- gene_biotypes[gene_biotypes$gene_biotype == "protein_coding", ]
write.csv(coding_genes, "./preprocessed_data/protein_coding_genes.csv", 
          row.names = FALSE)

res_coding <- res[rownames(res) %in% coding_genes$ensembl_gene_id, ]
res_coding_df <- as.data.frame(res_coding)
res_coding_df$ensembl_gene_id <- rownames(res_coding_df)
res_coding_df <- merge(res_coding_df, 
                       coding_genes[, c("ensembl_gene_id", "hgnc_symbol")], 
                       by = "ensembl_gene_id")
write.csv(res_coding_df, "./preprocessed_data/deseq2_results_protein_coding.csv", 
          row.names = FALSE)
```

### 4. Pathway Enrichment Analysis

```r
library(EnhancedVolcano)
library(clusterProfiler)
library(dplyr)
library(tibble)
library(readr)
library(enrichplot)
library(igraph)       
library(ggraph)
library(org.Hs.eg.db)
library(ggplot2)
library(GOplot)

res <- read.csv("./preprocessed_data/deseq2_results_protein_coding.csv", 
                check.names = FALSE)
padj_thresh <- 0.05
logfc_thresh <- 1

EnhancedVolcano(res,
                lab = res$hgnc_symbol,
                x = 'log2FoldChange',
                y = 'pvalue',
                pCutoff = 0.05,
                FCcutoff = 1.5,
                colAlpha = 0.4,
                title = NULL,
                legendLabels = c(),
                legendPosition = 'none',
                subtitle = NULL)

deg_sig <- res %>%
  filter(padj < 0.05, abs(log2FoldChange) > 1) %>%
  pull(hgnc_symbol) %>%
  na.omit()

ego <- enrichGO(gene = deg_sig,
                OrgDb = org.Hs.eg.db,
                keyType = "SYMBOL",
                ont = "BP",
                pAdjustMethod = "BH",
                pvalueCutoff = 0.05,
                readable = TRUE)

barplot(ego, showCategory = 20)
dotplot(ego, showCategory = 15)
cnetplot(ego, showCategory = 5, circular = TRUE, colorEdge = TRUE)

ego_df <- as.data.frame(ego)
ego_df <- ego_df %>%
  mutate(core_genes = gsub("/", ",", geneID)) %>%
  dplyr::select(ID, Description, GeneRatio, BgRatio, pvalue, p.adjust, 
                qvalue, Count, core_genes)
write_csv(ego_df, "./preprocessed_data/enrichGO_pathways_with_genes.csv")

gene_list <- res %>%
  filter(
    !is.na(padj),
    !is.na(log2FoldChange),
    !is.na(hgnc_symbol),
    hgnc_symbol != ""      
  ) %>%
  mutate(stat = -log10(padj) * sign(log2FoldChange)) %>%
  distinct(hgnc_symbol, .keep_all = TRUE) %>%
  arrange(desc(stat)) %>%
  dplyr::select(hgnc_symbol, stat) %>%
  tibble::deframe()

gsea_res <- gseGO(geneList = gene_list,
                  OrgDb = org.Hs.eg.db,
                  ont = "BP",
                  keyType = "SYMBOL",
                  nPerm = 10000,
                  minGSSize = 10,
                  maxGSSize = 500,
                  pvalueCutoff = 0.05,
                  verbose = TRUE,
                  pAdjustMethod = "BH")

gsea_df <- as.data.frame(gsea_res)
top_pathways <- gsea_df %>%
  arrange(p.adjust) %>%
  head(30)

top_pathways_clean <- top_pathways %>%
  mutate(core_enrichment_genes = strsplit(core_enrichment, "/")) %>%
  rowwise() %>%
  mutate(core_enrichment_genes_str = paste(core_enrichment_genes, collapse = ",")) %>%
  ungroup()

output_df <- top_pathways_clean %>%
  dplyr::select(ID, Description, NES, pvalue, p.adjust, core_enrichment_genes_str)
write_csv(output_df, "./preprocessed_data/top_gsea_pathways_with_genes.csv")

dotplot(gsea_res, showCategory = 20, split = ".sign") + 
  ggplot2::facet_grid(.~.sign)

gsea_res_sim <- pairwise_termsim(gsea_res)
emapplot(gsea_res_sim, showCategory = 30)
```


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
