
library(EnhancedVolcano)
library(clusterProfiler)
library(dplyr)
library(tibble)
library(readr)
library(enrichplot)
library(igraph)       
library(ggraph)
library(org.Hs.eg.db)
library(tibble)
library(ggplot2)
library(GOplot)


res <- read.csv("./preprocessed_data/deseq2_results_percentile_protein_coding.csv", check.names = FALSE)
padj_thresh <- 0.05
logfc_thresh <- 1

num_genes_tested <- sum(!is.na(res$padj))

num_degs <- sum(res$padj < padj_thresh, na.rm = TRUE)

num_samples <- ncol(dds)

summary_table <- data.frame(
  Num_Samples = num_samples,
  Genes_Tested = num_genes_tested,
  DEGs_Significant = num_degs,
  padj_Threshold = padj_thresh,
  log2FC_Threshold = logfc_thresh
)

print(summary_table)


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
                subtitle = NULL
                
)


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

head(ego@result$geneID)
head(deg_sig)
head(res$hgnc_symbol)

geneList <- res %>%
  dplyr::filter(!is.na(hgnc_symbol), !is.na(log2FoldChange)) %>%
  dplyr::select(hgnc_symbol, log2FoldChange) %>%
  tibble::deframe()

barplot(ego, showCategory = 20)
dotplot(ego, showCategory = 15)
cnetplot(ego, showCategory = 5, circular = TRUE, colorEdge = TRUE)
cnetplot(ego, foldChange=geneList, circular = TRUE, colorEdge = TRUE, layout = "fr",) 
heatplot(ego, showCategory = 10, foldChange=geneList) 


upsetplot(ego)
p <- upsetplot(ego)
p + theme_minimal() +
  theme(
    text = element_text(size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  ) +
  ggtitle("UpSet Plot of GO Enrichment - TCGA PRAD")



ego_df <- as.data.frame(ego)
ego_df <- ego_df %>%
  mutate(core_genes = gsub("/", ",", geneID)) %>%
  dplyr::select(ID, Description, GeneRatio, BgRatio, pvalue, p.adjust, qvalue, Count, core_genes)

write_csv(ego_df, "./preprocessed_data/enrichGO_pathways_with_genes_percentile_new.csv")

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
write_csv(output_df, "./preprocessed_data/top_gsea_pathways_with_genes_percentile_new.csv")

dotplot(gsea_res, showCategory = 20, split = ".sign") + ggplot2::facet_grid(.~.sign)

gsea_res_sim <- pairwise_termsim(gsea_res)
emapplot(gsea_res_sim, showCategory = 30)


