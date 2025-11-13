import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from tqdm import tqdm
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
import glob
import argparse
import seaborn as sns

potential_features_to_use = [
    'max_betweenness',
    'tas_pos_high_percent',
    'tas_pos_percent',
    'tas_neg_percent',
    'patch_density',
    'laplace_mean',
    'laplace_std',
    'num_hotspots',
    'avg_prob_in_region',
    'std_contrast',
    'percent_hotspot',
    'clustering_coeff',
    'avg_shortest_path',
    'min_dist',
    'ratio_of_cancer_to_tissue',
    'stroma_to_epithelial_ratio','region_axis_major_len',
    'region_eccentricity',  'region_perimeter', 'region_solidity'
]

normalization_method = {
    'max_betweenness': 'YJ',
    'tas_pos_high_percent': 'YJ',
    'tas_pos_percent': 'YJ',
    'tas_neg_percent': 'standard',
    'patch_density': 'YJ',
    'laplace_mean': 'standard',
    'laplace_std': 'YJ',
    'num_hotspots': 'YJ',
    'avg_prob_in_region': 'standard',
    'std_contrast': 'standard',
    'percent_hotspot': 'YJ',
    'clustering_coeff': 'YJ',
    'avg_shortest_path': 'YJ',
    'min_dist': 'YJ',
    'stroma_to_epithelial_ratio': 'YJ',
    'ratio_of_cancer_to_tissue': 'YJ',
    'region_perimeter': 'YJ',
    'region_eccentricity': 'YJ',
    'region_axis_major_len': 'YJ',
    'region_solidity': 'standard'
    
}

def plot_dist(features_to_use, normalization_method, all_features, output_dir):
    for feature in features_to_use:
        feat = all_features[feature].values
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
            

        new_name = f'{feature}_scaled'
        all_features[new_name] = feat_transformed
        sns.histplot(feat_transformed, kde = True)
        plt.savefig(os.path.join(output_dir, f'{feature}_scaled_dist.png'))

        return all_features

def feature_bins(df,feat_names):
    df = df.copy()
    
    for name in feat_names:
        if name == 'laplace_mean_scaled':
            df[f'{name}_q'] = pd.qcut(df[name], q=3, labels=[0, 1, 2])
        else:
            df[f'{name}_q'] = pd.qcut(df[name], q=2, labels=[0, 1])
    return df


def main(args):
    all_feats = pd.read_csv(args.dataset_root)
    all_feats = plot_dist(potential_features_to_use,
        normalization_method,
        all_feats,
        args.output_dir)
    new_names = [f'{name}_scaled' for name in features_to_use]
    all_feats = feature_bins(all_feats, new_names)
    all_feats.to_csv(args.dataset_root)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root', type = str
    )
    parser.add_argument(
        '--output_dir', type = str
    )

    args = parser.parse_args()
    main(args)