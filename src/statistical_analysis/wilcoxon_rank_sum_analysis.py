import os
import numpy as np
import pandas as pd
import glob
import pickle as pkl
from scipy.stats import mannwhitneyu
import scipy.stats as stats

from scipy.stats import shapiro, kstest
from scipy.stats import levene

from statsmodels.stats.multitest import multipletests

from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

import scipy
from math import e
from normalization_slide_level_feats import normalization_method
from feature_names import *
import itertools
import argparse

def scale_feats(all_feats):
    all_feats_scaled = {}
    for feature, norm_method in normalization_method.items():
        try:
            feat = all_feats[feature].values
            norm_method = normalization_method[feature]

            if norm_method == 'YJ':
                scaler = PowerTransformer(method='yeo-johnson')
                feat = feat.reshape(-1, 1)
                feat_transformed = scaler.fit_transform(feat)
                scaler = StandardScaler()
                feat_transformed = feat_transformed.reshape(-1, 1)
                feat_transformed = scaler.fit_transform(feat_transformed)

            elif norm_method == 'log':
                feat_transformed = np.log1p(feat)
                scaler = StandardScaler()
                feat_transformed = feat_transformed.reshape(-1, 1)
                feat_transformed = scaler.fit_transform(feat_transformed)

            elif norm_method == 'standard':
                scaler = StandardScaler()
                feat = feat.reshape(-1, 1)
                feat_transformed = scaler.fit_transform(feat)

            all_feats_scaled[feature] = feat_transformed.squeeze()
        except Exception as e:
            print(feature, e)

    all_feats_scaled['slide_name'] = all_feats['slide_name'].values
    all_feats_scaled['dominant'] = all_feats['dominant'].values
    all_feats_scaled['dominant_size'] = all_feats['dominant_size'].values
    return pd.DataFrame(all_feats_scaled)


def main(args):
    all_feats = pd.read_csv(args.all_feats, index_col = 0)
    all_feats = scale_feats(all_feats)

    all_feature_names = features_basic 
    all_feaure_names.extend(features_hotspots)
    all_feature_names.extend(features_graph)
    all_feature_names.extend(features_dist)

    low_grade_vs_high_grade = []

    for feature in all_feature_names:
        low = group_low_grade[[feature]]
        high = group_high_grade[[feature]]
        u, p = mannwhitneyu(low, high, alternative='two-sided')
        low_grade_vs_high_grade.append([feature, p[0]])

    low_grade_vs_high_grade = pd.DataFrame(low_grade_vs_high_grade, columns = ['feature', 'pvalue']) 
    mc_pval = multipletests(low_grade_vs_high_grade['pvalue'].values, alpha=0.05, method='bonferroni', maxiter=1, is_sorted=False, returnsorted=False)
    low_grade_vs_high_grade['adj_pvalue'] = mc_pval[1]
    low_grade_vs_high_grade.to_csv(os.path.join(args.save_root, 'low_grade_high_grade_wilcoxon.csv'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--slide_level_features', type = str
    )
    args = parser.parse_args()
    main(args)