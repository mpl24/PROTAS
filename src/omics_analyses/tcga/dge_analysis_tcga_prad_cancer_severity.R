library(DESeq2)

counts <- read.csv("./preprocessed_data/expression_matrix.csv", check.names = FALSE)
meta <- read.csv("./preprocessed_data/tcga_prad_metadatagrade_group.csv")
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
  design = ~ grade_group_mapped  
)

dds <- DESeq(dds)
res <- results(dds) 
write.csv(as.data.frame(res), "./preprocessed_data/deseq2_results_new_GG.csv")

library(biomaRt)
mart <- biomaRt::useMart(biomart = "ENSEMBL_MART_ENSEMBL", dataset = "hsapiens_gene_ensembl")

gene_symbols <- biomaRt::getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"),
                               filters = "ensembl_gene_id",
                               values = ensembl_ids,
                               mart = mart)

write.csv(as.data.frame(gene_symbols), "./preprocessed_data/gene_symbols_new_GG.csv")

gene_biotypes <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol", "gene_biotype"),
                       filters = "ensembl_gene_id",
                       values = ensembl_ids,
                       mart = ensembl)

coding_genes <- gene_biotypes[gene_biotypes$gene_biotype == "protein_coding", ]

write.csv(coding_genes, "./preprocessed_data/protein_coding_genes_GG.csv", row.names = FALSE)

res_coding <- res[rownames(res) %in% coding_genes$ensembl_gene_id, ]

res_coding_df <- as.data.frame(res_coding)
res_coding_df$ensembl_gene_id <- rownames(res_coding_df)
res_coding_df <- merge(res_coding_df, coding_genes[, c("ensembl_gene_id", "hgnc_symbol")], by = "ensembl_gene_id")

write.csv(res_coding_df, "./preprocessed_data/deseq2_results_coding_only_GG.csv", row.names = FALSE)

