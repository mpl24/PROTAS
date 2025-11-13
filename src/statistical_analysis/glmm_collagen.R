
library(lme4)
library(broom.mixed)
library(dplyr)
library(bestNormalize)
library(readr)

merged_df <- read_csv('PATH TO COLLAGEN FEATURES')
norm_methods <- read_csv('PATH TO NORMALIZATION METHOD FOR EACH COLLAGEN FEATURE')

all_results <- list()
all_summaries <- list()

for (i in seq_len(nrow(norm_methods))) {
  feat <- norm_methods$feature[i]
  norm_method <- norm_methods$norm_method[i]
  
  message(sprintf("Running GLMM for feature: %s with normalization: %s", feat, norm_method))
  
  subset_df <- merged_df[, c("slide_name", "tas_label", feat)]
  colnames(subset_df)[3] <- "feature"  
  subset_df$patch_label <- as.integer(subset_df$tas_label)
  
  if (norm_method == "log") {
    subset_df$feature[subset_df$feature < 0] <- 0
    subset_df$feature <- log1p(subset_df$feature)
  } else if (norm_method == "standard") {
    subset_df$feature <- scale(subset_df$feature, center = TRUE, scale = TRUE)
  } else if (norm_method == "min_max") {
    min_val <- min(subset_df$feature, na.rm = TRUE)
    max_val <- max(subset_df$feature, na.rm = TRUE)
    subset_df$feature <- (subset_df$feature - min_val) / (max_val - min_val)
  } else if (norm_method == "YJ") {
    yj <- yeojohnson(subset_df$feature, standardize = TRUE)
    subset_df$feature <- predict(yj)
  } else if (norm_method == "z-score") {
    subset_df$feature <- scale(subset_df$feature)
  }
  
  model <- glm(patch_label ~ feature + as.factor(slide_name),
               data = subset_df, family = binomial(link = "logit"))
  
  summary_df <- tidy(model, conf.int = TRUE)  
  summary_df$Feature <- feat
  all_summaries[[length(all_summaries) + 1]] <- summary_df
  
  feature_pval <- summary_df$p.value[summary_df$term == "feature"]
  message(sprintf("Feature: %s | p-value: %.3g", feat, feature_pval))
  
  all_results[[feat]] <- coef(model)
}

results_df <- do.call(rbind, all_summaries)
write_csv(results_df, "glmm_results_df_subset_08_05_COLLAGEN.csv")
