import os
import pandas as pd
import numpy as np
import glob
import pickle as pkl
from tqdm import tqdm
import argparse

def set_up_rs_mean_prob_groups(args, metadata, rs_patient_df):

    q75_rs = rs_patient_df["rs_score"].quantile(0.75)
    rs_patient_df["group_percentile"] = (rs_patient_df["rs_score"] >= q75_rs).astype(int)

    raw_expression = pd.read_csv(args.raw_expression_data)
    metadata = metadata[metadata['sample_type'] == 'Primary Tumor']
    metadata = metadata.merge(rs_patient_df, on = 'patient', how = 'right')

    sample_ids = metadata['sample_id'].unique()
    raw_counts = raw_expression[sample_ids]

    raw_counts.to_csv(os.path.join(args.save_root, 'expression_matrix_mean_prob_rs.csv'))
    metadata.to_csv(os.path.join(args.save_root, 'metadata_mean_prob_rs.csv'))

def set_up_gg_groups(tcga_prad_clin_matrix, args, metadata):
    metadata = metadata.reset_index(drop = True)

    metadata_sample_cleaned = []
    for ind, row in metadata.iterrows():
        sample = str(row['sample'])
        sample = sample[:-1]
        metadata_sample_cleaned.append(sample)

    metadata['sample'] = metadata_sample_cleaned
    tcga_prad_clin_matrix = tcga_prad_clin_matrix.rename(columns = {
        'sample_id_info': 'sample',
    })
    metadata = metadata[metadata['sample_type'] == 'Primary Tumor']
    metadata_with_gleason = metadata.merge(tcga_prad_clin_matrix, on = ['patient', 'sample'], how = 'left')

    mapping_grade_group = {
        1: 0,
        2: 0,
        3: 1,
        4: 1,
        5: 1    
    }
    metadata_with_gleason['grade_group_mapped'] = metadata_with_gleason['grade_group'].map(mapping_grade_group)
    raw_expression = pd.read_csv(args.raw_expression_data)

    sample_ids = metadata_with_gleason['sample_id'].unique()
    raw_counts = raw_expression[sample_ids]

    raw_counts.to_csv(os.path.join(args.save_root, 'expression_matrix_grade_group.csv'))
    metadata_with_gleason.to_csv(os.path.join(args.save_root, 'metadata_grade_group.csv'))


def preprocess(args, tcga_names, slide_level_rs_features):
    metadata = pd.read_csv(args.metadata, index_col = 0)
    tcga_prad_clin_matrix = pd.read_csv(args.tcga_prad_clinical_matrix, sep = '\t')
    tcga_prad_clin_matrix = tcga_prad_clin_matrix[['primary_pattern', 'secondary_pattern', 'sampleID', '_PATIENT']]
    tcga_prad_clin_matrix = tcga_prad_clin_matrix.reset_index(drop = True)

    grade_groups = []

    for ind, row in tcga_prad_clin_matrix.iterrows():
        prim_pattern = row.primary_pattern
        secondary_pattern = row.secondary_pattern
        if prim_pattern == 3:
            if secondary_pattern == 4:
                gg = 2
            elif secondary_pattern == 5:
                gg = 4
            elif secondary_pattern == 3:
                gg = 1
                
        if prim_pattern == 4:
            if secondary_pattern == 3:
                gg = 3
            elif secondary_pattern == 4:
                gg = 4
            elif secondary_pattern == 5:
                gg = 5
        
        if prim_pattern == 5:
            if secondary_pattern == 3:
                gg = 4
            elif secondary_pattern == 4:
                gg = 5
            elif secondary_pattern == 5:
                gg = 5
            
        grade_groups.append(gg)
    
    tcga_prad_clin_matrix['grade_group'] = grade_groups
    tcga_prad_clin_matrix = tcga_prad_clin_matrix.rename(columns = {
        'sampleID': 'sample_id_info',
        '_PATIENT': 'patient'
    })

    rs_patient_df = (
        slide_level_rs_features
        .groupby("patient")["rs_pos_percent"]
        .mean()
        .reset_index()
        .rename(columns={"rs_pos_percent": "rs_score"})
    )

    return rs_patient_df, metadata, tcga_prad_clin_matrix


def main(args):
    slide_level_rs_features = pd.read_csv(args.slide_level_rs_features, index_col = 0)
    tcga_names = slide_level_rs_features['slide_name'].values

    rs_patient_df, metadata, tcga_prad_clin_matrix = preprocess(args, tcga_names, slide_level_rs_features)

    set_up_rs_mean_prob_groups(args, metadata, rs_patient_df)
    set_up_gg_groups(tcga_prad_clin_matrix, args, metadata)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_root', type = str,
        desc = 'Where to save the processed metadata and expression data'
    )
    parser.add_argument(
        '--tcga_prad_metadata',type = str,
        desc = 'TCGA PRAD metadata containing barcode, patient, sample, etc'
    )
    parser.add_argument(
        '--tcga_prad_clinical_matrix', type = str
        desc = 'UCSC Xena Clincal Matrix'
    )
    parser.add_argument(
        '--slide_level_rs_features', type = str,
        desc = 'Slide level RS features extracted for tcga-prad WSI samples'
    )
    parser.add_argument(
        '--raw_expression_data', type = str,
        desc = 'Raw expression matrix in csv format'
    )

    args = parser.parse_args()
    main(args)