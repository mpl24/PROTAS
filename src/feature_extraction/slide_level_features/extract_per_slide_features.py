import os
import numpy as np
import pandas as pd
from features_functions import *
from tqdm import tqdm
from grade_mappings import *
import argparse

from scipy.stats import rankdata


def set_up_data(slide_name, args, optimal_cutoff = 0.5):
    model_preds = pd.read_csv(os.path.join(args.dataframe_root, f'{slide_name}.csv'),index_col = 0)
    model_preds['rs_pred'] = (model_preds['mean_pred'] > optimal_cutoff).astype(int)
    model_preds_no_cancer = model_preds[model_preds['in_cancer'] == 0]
    confident_set = model_preds_no_cancer[model_preds_no_cancer['confident'] == 1]
    # confident set is just the patches outside of cancer that are also confident preds
    return model_preds, confident_set, model_preds_no_cancer 

def main(args):

    os.makedirs(args.save_root, exist_ok = True)
    full_dataset_clin = pd.read_csv(args.clinical_data, index_col = 0)
    gg_per_slide = full_dataset_clin[['slide_name', 'gleason_score', 'grade_group']]

    slide_names = list(full_dataset_clin['slide_name'].unique())

    basic_stats_all = []

    connected_comp_size = 3
    hotspot_feats_all = []

    distance_feats_all = []

    centroids_dict = {}
    region_probs_dict = {}
    region_patch_counts_dict = {}

    basic_stats_all = []

    connected_comp_size = 3
    hotspot_feats_all = []

    centroids_dict = {}
    region_probs_dict = {}
    region_patch_counts_dict = {}

    distance_feats_all = []

    all_topo_feats = []



    for slide_name in tqdm(slide_names):
        current_slide = confident_set[confident_set['slide_name'] == slide_name].copy()
        current_all_preds = model_preds_no_cancer[model_preds_no_cancer['slide_name'] == slide_name].copy()
        
        ## BASIC FEATURES
        basic_stats_feats = calculate_basic_stats(current_slide, current_all_preds)
        basic_stats_feats['slide_name'] = slide_name
        basic_stats_all.append(basic_stats_feats)
        
        
        ## HOTSPOT FEATURES
        features, region_patch_counts, region_probs, centroids = full_slide_features(
            current_slide, len(current_all_preds), connected_comp_size)
        features['slide_name'] = slide_name
        hotspot_feats_all.append(features)
        centroids_dict[slide_name] = centroids
        region_probs_dict[slide_name] = region_probs
        region_patch_counts_dict[slide_name] = region_patch_counts
        
        ## DISTANCE FEATURES
        distance_feats = extract_distance_feats(confident_set, distance_col = 'distance_from_cancer_mm')
        distance_feats['slide_name'] = slide_name
        distance_feats_all.append(distance_feats)
        
        ## GRAPH FEATURES
        topo_feats = get_graph_feats(confident_set, centroids_dict[slide_name])
        topo_feats['slide_name'] = slide_name
        all_topo_feats.append(topo_feats)

    basic_stats_all = pd.DataFrame(basic_stats_all)
    hotspot_feats_all = pd.DataFrame(hotspot_feats_all)
    distance_feats_all = pd.DataFrame(distance_feats_all)
    all_topo_feats = pd.DataFrame(all_topo_feats)

    all_feats = basic_stats_all.merge(hotspot_feats_all, on = 'slide_name', how = 'outer')
    all_feats = all_feats.merge(distance_feats_all, on = 'slide_name', how = 'outer')
    all_feats = all_feats.merge(all_topo_feats, on = 'slide_name', how = 'outer')
    all_feats = all_feats.merge(gg_per_slide, on = 'slide_name', how = 'left')

    all_feats.to_csv(os.path.join(args.save_root, 'all_features.csv'))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataframe_root', type = str, default = ''
    )
    parser.add_argument(
        '--clinical_data', type = str, default = ''
    )
    parser.add_argument(
        '--save_root', type = str, default = ''
    )
    args = parser.parse_args()
    main(args)