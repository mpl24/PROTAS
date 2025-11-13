import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import argparse

def smooth_labels(dataset):
    confidence_manual = []
    for ind, row in tqdm(dataset.iterrows()):
        grade = row['grade']
        label = row['label']
        if grade == '3+3'  and label == 0:
            confidence_manual.append(0.85)
        elif grade == '3+4' and label == 0:
            confidence_manual.append(0.5)
        elif grade == '3+4' and label == 1:
            confidence_manual.append(0.5)
        elif grade == '4+3' and label == 1:
            confidence_manual.append(0.6)
        elif grade == '4+4' and label == 1:
            confidence_manual.append(0.7)
        elif grade == '3+5' and label == 1:
            confidence_manual.append(0.6)
        elif grade in ['4+5', '5+4', '5+5'] and label == 1:
            confidence_manual.append(0.85)
        else:
            confidence_manual.append(None)

    dataset['confidence_score_prior'] = confidence_manual
    binary_labels = dataset.label.values
    confidences = dataset.confidence_score_prior.values
    smoothed_labels_confidence_prior = binary_labels * confidences + (1-binary_labels)*(1-confidences)
    dataset['smoothed_label_confidence_prior'] = smoothed_labels_confidence_prior
    return dataset


def main(args):

    dataset = pd.read_csv(args.dataset_csv, index_col = 0)
    dataset = smooth_labels(dataset)
    
    file_name = args.dataset_csv.split('/')[-1].split('.')[0]
    save_root = args.dataset_csv.split('/')[:-1]
    save_root = '/'.join(save_root)
    file_name_new = f'{file_name}_smoothed_labels'
    dataset.to_csv(os.path.join(
        save_root, f'{file_name_new}.csv'
    ))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_csv', type = str
    )
    args = parser.parse_args()
    main(args)