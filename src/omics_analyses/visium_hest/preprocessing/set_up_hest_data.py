import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import anndata
from anndata import AnnData
from pathlib import Path
import scipy.io as sio
import scanpy as sc
from tqdm import tqdm
import argparse

def preprocess_visium_data(args, prostate_sample_slides):
    save_folder = os.path.join(args.save_root, 'sub_patches_min_counts_50_min_cells_5')
    os.makedirs(save_folder, exist_ok = True)

    for name in prostate_sample_slides:
        data = f'/raid/HEST-1K/st/{name}.h5ad'
        adata = sc.read_h5ad(data)
        sc.pp.filter_cells(adata, min_counts = 50)
        sc.pp.filter_genes(adata, min_cells = 5)
        sc.pp.normalize_total(adata, target_sum = 1e4)
        sc.pp.log1p(adata)

        counts_df = pd.DataFrame(
            adata.X.toarray(),
            index=adata.obs_names,
            columns=adata.var_names
        )
        coords_df = pd.DataFrame(
            adata.obsm["spatial"],
            index=adata.obs_names,
            columns=["x", "y"]
        )
        
        nUMI_df = pd.DataFrame(
            counts_df.sum(axis=1),
            columns=["nUMI"]
        )
        coords_df.to_csv(os.path.join(save_folder, f'coords_{name}.csv'))
        nUMI_df.to_csv(os.path.join(save_folder, f'nUMI_{name}.csv'))
        counts_df.to_csv(os.path.join(save_folder, f'counts_{name}.csv'))

    return save_folder
        
def filter_to_rs_preds(consolidated_rs_preds, data_root):
    all_counts_files = sorted(glob(os.path.join(data_root, f'counts*.csv')))
    for file in all_counts_files:
        name = file.split('/')[-1].split('.')[0].split('_')[1]
        curr_counts_data = pd.read_csv(file)
        
        curr_counts_data = curr_counts_data.rename(columns = {'Unnamed: 0': 'barcode'})
        consoliated_preds_curr_slide = consolidated_rs_preds[consolidated_rs_preds['slide_name'] == name]

        counts_data_subset = curr_counts_data[curr_counts_data['barcode'].isin(consoliated_preds_curr_slide['barcode'])].reset_index(drop = True)
        consoliated_preds_curr_slide = consoliated_preds_curr_slide[consoliated_preds_curr_slide['barcode'].isin(counts_data_subset['barcode'])]
        counts_data_subset.to_csv(os.path.join(data_root, f'counts_{name}.csv'))
        consoliated_preds_curr_slide.to_csv(os.path.join(data_root, f'filtered_metadata_{name}.csv'))
    

def assemble_metadata(data_root):
    assembled_metadata = []
    metadata_files = sorted(glob(os.path.join(data_root, f'filtered_metadata*.csv')))
    for file in metadata_files:
        curr_df = pd.read_csv(file, index_col = 0)
        assembled_metadata.append(curr_df)
        
    assembled_metadata = assembled_metadata.drop(columns = ['mean_pred', 'xloc', 'yloc'])
    assembled_metadata = assembled_metadata.reset_index(drop = True)
    assembled_metadata = assembled_metadata.rename(columns = {'slide_name': 'replicate', 'rs_prediction': 'rs_pred'})
    assembled_metadata.to_csv(os.path.join(data_root, 'metadata.csv'))

def filter_expression_data(data_root, prostate_sample_slides):
    expression_matrix = []
    gene_df = {}
    expression_files = sorted(glob(os.path.join(data_root, f'counts*.csv')))
    for file in tqdm(expression_files):
        name = file.split('/')[-1].split('_')[1].split('.')[0]
        curr_df = pd.read_csv(file, index_col = 0)
        gene_names = curr_df.columns[1:]
        gene_df[name] = list(gene_names)

    sets_of_genes = []
    for slide_name in prostate_sample_slides:
        sets_of_genes = set(gene_df[slide_name])
    
    overlap = sets_of_genes[0].intersection(sets_of_genes[1:])
    overlap = list(overlap)
    overlap.append('barcode')

    expression_matrix = []
    expression_files = sorted(glob(os.path.join(data_root, f'counts*.csv')))
    for file in tqdm(expression_files):
        name = file.split('/')[-1].split('_')[1].split('.')[0]
        curr_df = pd.read_csv(file, index_col = 0)
        curr_df = curr_df[overlap]
        expression_matrix.append(curr_df)

    expression_matrix = pd.concat(expression_matrix)
    cols = expression_matrix.columns.tolist()
    new_order = [cols[-1]] + cols[:-1]
    expression_matrix = expression_matrix[new_order]
    expression_matrix.to_csv(os.path.join(data_root, 'expression_matrix.csv'))

def filter_coords(data_root):

    coord_files = sorted(glob(os.path.join(data_root, 'coords*')))

    for file in tqdm(coord_files):
        name = file.split('/')[-1].split('.')[0].split('_')[1]
        metadata_file = os.path.join(data_root, f'filtered_metadata_{name}.csv')
        
        coords = pd.read_csv(file)
        coords = coords.rename(columns = {'Unnamed: 0': 'barcode'})
        coords['replicate'] = [name] * len(coords)
        meta = pd.read_csv(metadata_file, index_col = 0)
        coords = coords[coords['barcode'].isin(meta['barcode'])]
        coords = coords.reset_index(drop = True)
        coords.to_csv(os.path.join(data_root, f'coords_{name}.csv'))


def main(args):
    prostate_sample_slides = ['MEND151', 'MEND162', 'MEND60', 'MEND62']
    data_root = preprocess_visium_data(args, prostate_sample_slides)

    consoliated_rs_preds = pd.read_csv(args.stroma_RS_predictions, index_col = 0)
    filter_to_rs_preds(consolidated_rs_preds, data_root)
    assemble_metadata(data_root)
    filter_expression_data(data_root, prostate_sample_slides)
    filter_coords(data_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--h5ad_root'
    )
    parser.add_argument(
        '--stroma_RS_predictions', type =str, desc = ''
    )
    parser.add_argument(
        '--save_root', type = str, desc = ''
    )

    main(args)