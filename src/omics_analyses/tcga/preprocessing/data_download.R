library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
library(stringr)

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

##ESTIMATED TIME TO DOWNLOAD: 15 mins