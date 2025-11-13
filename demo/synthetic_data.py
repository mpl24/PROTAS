import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import binary_dilation, gaussian_filter
from PIL import Image
import json
import random
import argparse

class SyntheticDataGenerator:
    def __init__(self, save_root, seed = 24):
        self.save_root = save_root
        self.seed = seed
        np.random.seed(seed)

        self.patch_dir = os.path.join(self.save_root, 'patches')
        self.feature_dir = os.path.join(self.save_root, 'uni_features')
        self.cancer_mask_dir = os.path.join(self.save_root, 'cancer_masks')
        self.tissue_mask_dir = os.path.join(self.save_root, 'tissue_masks')
        self.output_dir = os.path.join(self.save_root, 'models')

        os.makedirs(self.patch_dir, exist_ok = True)
        os.makedirs(self.feature_dir, exist_ok = True)
        os.makedirs(self.cancer_mask_dir, exist_ok = True)
        os.makedirs(self.tissue_mask_dir, exist_ok = True)
        os.makedirs(self.output_dir, exist_ok = True)


    def generate_prostate_mask(self, width = 2000, height = 2000):
        mask = np.zeros((height, width), dtype = np.uint8)
        center_x = width//2
        center_y = height//2

        radius_x = np.random.randint(600, 800)
        radius_y = np.random.randint(500, 650)

        y_coords, x_coords = np.ogrid[:height, :width]

        ellipse = ((x_coords - center_x) ** 2/radius_x ** 2 + (y_coords - center_y) **2 / radius_y ** 2) <= 1

        mask[ellipse] = 1

        mask = gaussian_filter(mask.astype(float), sigma = 5) > 0.5
        return mask.astype(np.uint8)

    def generate_cancer_mask(self, tissue_mask, location = 'random'):
        height, width = tissue_mask.shape
        cancer_mask = np.zeros_like(tissue_mask)

        y_coords, x_coords = np.where(tissue_mask > 0)
        tissue_center_x = int(np.mean(x_coords))
        tissue_center_y = int(np.mean(y_coords))

        locations = {
            'left_apex': (tissue_center_x - width // 5, tissue_center_y - height // 4),
            'right_apex': (tissue_center_x + width // 5, tissue_center_y - height // 4),
            'left_base': (tissue_center_x - width // 5, tissue_center_y + height // 4),
            'right_base': (tissue_center_x + width // 5, tissue_center_y + height // 4),
            'left_mid': (tissue_center_x - width // 4, tissue_center_y),
            'right_mid': (tissue_center_x + width // 4, tissue_center_y),
        }

        if location == 'random':
            location = np.random.choice(list(locations.keys()))

        cancer_center_x, cancer_center_y = locations[location]
        cancer_center_x += np.random.randint(-100, 100)
        cancer_center_y += np.random.randint(-100, 100)

        cancer_size_category = np.random.choice(['small', 'medium', 'large'], p=[0.3, 0.5, 0.2])

        if cancer_size_category == 'small':
            radius = np.random.randint(100, 200)
        elif cancer_size_category == 'medium':
            radius = np.random.randint(200, 350)
        else:  # large
            radius = np.random.randint(350, 500)

        y_grid, x_grid = np.ogrid[:height, :width]

        radius_x = radius * np.random.uniform(0.8, 1.2)
        radius_y = radius * np.random.uniform(0.8, 1.2)

        cancer_region = ((x_grid - cancer_center_x) ** 2 / radius_x ** 2 + 
                        (y_grid - cancer_center_y) ** 2 / radius_y ** 2) <= 1

        cancer_mask = cancer_region & (tissue_mask > 0)

        cancer_mask = gaussian_filter(cancer_mask.astype(float), sigma=3) > 0.4
        cancer_mask = cancer_mask & (tissue_mask > 0)

        return cancer_mask.astype(np.uint8)

    def generate_patches_for_slide(
        self, 
        slide_name,
        tissue_mask,
        cancer_mask,
        n_patches, 
        patch_size = 256
    ):
        tissue_mask = np.array(tissue_mask)
        cancer_mask = np.array(cancer_mask)
        mask_height, mask_width = tissue_mask.shape

        slide_width = np.random.randint(50000, 75001)
        slide_height = 50000

        scale_x = slide_width / mask_width
        scale_y = slide_height / mask_height

        patches = []

        mask_y_coords, mask_x_coords = np.where(tissue_mask > 0)

        patch_indices = np.random.choice(len(mask_y_coords), size = n_patches, replace = True)

        pixels_per_mm = 1000

        for idx in tqdm(patch_indices):
            stroma_label = 1 if random.random() < 0.4 else 0

            mask_y_center = mask_y_coords[idx]
            mask_x_center = mask_x_coords[idx]

            x_center = int(mask_x_center * scale_x)
            y_center = int(mask_y_center * scale_y)

            jitter_range = int(min(scale_x, scale_y)/2)
            x_center += np.random.randint(-jitter_range, jitter_range + 1)
            y_center += np.random.randint(-jitter_range, jitter_range + 1)

            x_loc = x_center - patch_size // 2
            y_loc = y_center - patch_size // 2

            x_loc = max(0, min(x_loc, slide_width - patch_size))
            y_loc = max(0, min(y_loc, slide_height - patch_size))

            mask_x_check = int(x_center / scale_x)
            mask_y_check = int(y_center / scale_y)

            mask_x_check = max(0, min(mask_x_check, mask_width - 1))
            mask_y_check = max(0, min(mask_y_check, mask_width - 1))

            in_cancer = int(cancer_mask[mask_y_check, mask_x_check] > 0)

            if in_cancer:
                distance_mm_bin = 0
            else:
                cancer_coords = np.argwhere(cancer_mask > 0)
                if len(cancer_coords) > 0:
                    distance = np.sqrt(
                        (cancer_coords[:, 0] - mask_y_check) ** 2 +
                        (cancer_coords[:, 0] - mask_x_check) ** 2
                    )
                    min_dist_mask_pixels = np.min(distance)

                    min_dist_slide_pixels = min_dist_mask_pixels * np.mean([scale_x, scale_y])
                    distance_mm = min_dist_slide_pixels / pixels_per_mm
                    distance_mm_bin = int(np.floor(distance_mm) + 1)
                else:
                    distance_mm_bin = None

            patches.append({
                'slide_name': slide_name,
                'x_loc': int(x_loc),
                'y_loc': int(y_loc),
                'in_cancer': in_cancer,
                'distance_from_cancer_mm': distance_mm_bin,
                'stroma_label': stroma_label
            })
        return patches, slide_width, slide_height

    def generate_uni_features(self, n_features):
        features = np.random.randn(n_features, 1024).astype(np.float32)
        return features

    def generate_clinical_data(self, patient_slides):
        clinical_data = []

        for patient_id, slide_info in tqdm(patient_slides.items(), total = len(patient_slides)):
            slide_name = slide_info['slide_name']
            age = np.clip(np.random.normal(65, 8), 45, 85)
            
            gleason_primary = np.random.choice([3, 4, 5], p = [0.4, 0.4, 0.2])
            gleason_secondary = np.random.choice([3, 4, 5], p = [0.4, 0.4, 0.2])

            gleason_score = gleason_primary + gleason_secondary

            if gleason_score <= 6:
                grade_group = 1
            elif gleason_score == 7 and gleason_primary == 3:
                grade_group = 2
            elif gleason_score == 7 and gleason_primary == 4:
                grade_group = 3
            elif gleason_score == 8:
                grade_group = 4
            else:
                grade_group = 5

            
            clinical_data.append({
                'patient_id': patient_id,
                'slide_name': slide_name,
                'age': age,
                'gleason_score': gleason_score,
                'gleason_primary': gleason_primary,
                'gleason_secondary': gleason_secondary,
                'grade_group': grade_group,
            })

        return pd.DataFrame(clinical_data)

    def generate_all(self, n_patients = 30):
        all_patches = []
        patient_slides = {}

        for patient_idx in tqdm(range(n_patients), desc = 'Generating patients'):
            patient_id = f'synthetic_p{patient_idx}'
            slide_name = f'synthetic_slide_{patient_idx}'

            n_patches = np.random.randint(5000, 15001)
            tissue_mask = self.generate_prostate_mask()
            cancer_mask = self.generate_cancer_mask(tissue_mask)

            tissue_mask = Image.fromarray(tissue_mask)
            tissue_mask.save(os.path.join(self.tissue_mask_dir, f'{slide_name}.png'))

            cancer_mask = Image.fromarray(cancer_mask)
            cancer_mask.save(os.path.join(self.cancer_mask_dir, f'{slide_name}.png'))

            patches, slide_width, slide_height = self.generate_patches_for_slide(
                slide_name, tissue_mask, cancer_mask, n_patches
            )
            features = self.generate_uni_features(len(patches))

            feature_path_root = os.path.join(self.feature_dir, slide_name)
            os.makedirs(feature_path_root, exist_ok = True)
            for i, patch in enumerate(patches):
                feature_path = os.path.join(feature_path_root, f"{patch['x_loc']}_{patch['y_loc']}.npy")
                np.save(feature_path, features[i])

                patch['feature_path'] = feature_path
                all_patches.append(patch)

            patient_slides[patient_id] = {
                'slide_name':slide_name,
                'n_patches': len(patches),
                'slide_width': slide_width,
                'slide_height': slide_height
            }
        
        test_df = pd.DataFrame(all_patches)
        test_df.to_csv(os.path.join(self.save_root, 'synthetic_test_data.csv'))

        clinical_df = self.generate_clinical_data(patient_slides)
        clinical_df.to_csv(os.path.join(self.save_root, 'synthetic_clinical_data.csv'))


            


def main(args):
    os.makedirs(args.save_root, exist_ok = True)
    generator = SyntheticDataGenerator(
        save_root = args.save_root,
        seed = args.seed
    )
    generator.generate_all(n_patients = args.num_to_generate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type = int, default = 24
    )
    parser.add_argument(
        '--save_root', type = str, default = './demo/synthetic_data'
    )
    parser.add_argument(
        '--num_to_generate', type = int, default = 5
    )

    args = parser.parse_args()
    main(args)