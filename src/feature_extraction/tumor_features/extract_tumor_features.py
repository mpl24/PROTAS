import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import pickle as pkl
import seaborn as sns
from scipy.stats import mannwhitneyu
import scipy.stats as stats

import scipy
from math import e
import itertools

from tqdm import tqdm
import random

from PIL import Image
from skimage.transform import resize
from skimage.measure import label, regionprops, regionprops_table
from skimage import measure

def find_max_region(regions):
    max_area = 0
    max_ind = 0

    for ind, region in enumerate(regions):
        if region.area > max_area:
            max_ind = ind
            max_area = region.area
    
    return max_ind, max_area


def get_downsample(tissue_path, slide_path):
    slide = openslide.OpenSlide(slide_path)
    mask = np.array(Image.open(tissue_path))
    thumbnail = slide.get_thumbnail(mask.shape[::-1])
    downsample_rate = slide.level_dimensions[0][0] / mask.shape[1]
    
    return thumbnail, downsample_rate

def set_up_data(args):
    all_cancer_masks = sorted(
        glob.glob(
            os.path.join(
                args.cancer_mask_root, '*.png')))

    dataset = {}
    
    for cancer_mask in all_cancer_masks:
        slide_name = cancer_mask.split('/')[-1].split('.')[0]
        tissue_mask = os.path.join(args.tissue_mask_root, f'{slide_name}.png')
        dataset[slide_name] = {
            'cancer_mask': cancer_mask,
            'tissue_mask': tissue_mask
        }

    return dataset

def main(args):
    dataset = set_up_data(args)

    slide_names = dataset.keys()

    cancer_feats = []
    for slide_name in tqdm(slide_names):
        dataset = data_dict[slide_name]

        cancer_mask = np.array(Image.open(dataset['cancer_mask']))
        tissue_mask = np.array(Image.open(dataset['tissue_mask']))

        cancer_mask = resize(
                cancer_mask,
                tissue_mask.shape,
                mode="edge",
                anti_aliasing=False,
                anti_aliasing_sigma=None,
                preserve_range=True,
                order=0,
            )

        tissue_mask = tissue_mask.astype(bool)
        cancer_mask = cancer_mask.astype(bool)

        cancer_in_tissue = np.logical_and(tissue_mask, cancer_mask)
        percent_cancer_in_tissue = cancer_in_tissue.sum() / tissue_mask.sum()

        cancer_area = np.sum(cancer_mask)
        tissue_area = np.sum(tissue_mask)

        if tissue_area == 0:
            ratio = np.nan  
        else:
            ratio = cancer_area / tissue_area

        cancer_mask = cancer_mask.astype(np.uint8)
        labelled = measure.label(cancer_mask)
        regions = regionprops(labelled)
        max_ind, max_area = find_max_region(regions)
        region = regions[max_ind]
        max_region = np.where(labelled == region.label, 1, 0)

        stroma_info_within_cancer_csv = pd.read_csv(args.stroma_info_within_cancer_csv, index_col = 0)
        current_slide = stroma_info_within_cancer_csv[stroma_info_within_cancer_csv['slide_name'] == slide_name]

        stroma_percent_in_cancer = current_slide['stroma_patch_count'].values[0]/current_slide['all_patch_count'].values[0]        
        stroma_to_epithelial_ratio = current_slide['stroma_patch_count'].values[0]/current_slide['epithelial_patch_count'].values[0]

        cancer_feats.append([
            slide_name, 
            percent_cancer_in_tissue, 
            ratio, 
            stroma_percent_in_cancer, 
            stroma_to_epithelial_ratio,
            region.area,
            region.area_bbox,
            region.area_convex,
            region.axis_major_length,
            region.axis_minor_length,
            region.eccentricity,
            region.equivalent_diameter_area,
            region.feret_diameter_max,
            region.perimeter,
            region.solidity
            ])
        

    cancer_feats = pd.DataFrame(cancer_feats, columns = [
        'slide_name', 
        'percent_cancer_in_tissue', 
        'ratio_of_cancer_to_tissue', 
        'stroma_percent_within_cancer', 
        'stroma_to_epithelial_ratio',
        'region_area',
        'region_area_bbox',
        'region_area_convex',
        'region_axis_major_len',
        'region_axis_min_len',
        'region_eccentricity',
        'region_equivalent_diameter_area',
        'region_feret_diameter_max',
        'region_perimeter',
        'region_solidity'],)

    cancer_feats.to_csv(os.path.join(args.save_root, './cancer_features.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type = str)
    parser.add_argument("--cancer_mask_root", type = str)
    parser.add_argument("--tissue_mask_root", type = str)
    parser.add_argument("--stroma_info_within_cancer_csv", type = str)
    
    args = parser.parse_args()
    
    main(args)