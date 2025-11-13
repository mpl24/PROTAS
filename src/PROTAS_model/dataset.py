import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import logging
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate
import threading

class PrefetchLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.loader_iter = iter(dataloader)
        self.prefetched_batch = None
        self.done = False
        self.thread = threading.Thread(target = self._prefetch)
        self.thread.start()

    def __len__(self):
        return len(self.dataloader)

    def _prefetch(self):
        try:
            self.prefetched_batch = next(self.loader_iter)
        except StopIteration:
            self.done = True
            self.prefetched_batch = None
        except Exception as e:
            print(f"Error during prefetching: {e}")
            self.prefetched_batch = None
            self.done = True
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration
        while self.prefetched_batch is None: 
            self._prefetch()
            if self.done:
                raise StopIteration
        batch = self.prefetched_batch
        self._prefetch()
        return batch


def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return []
    return default_collate(batch)

def get_random_transform(uni_feature_root, uni_transform_feature_root, x, y, slide_name, num_versions = 50, percent_orig = 0.1):
    if random.random() < percent_orig:
        img = os.path.join(uni_feature_root, f'{slide_name}/{x}_{y}.npy')
    else:
        random_version = random.randint(1, num_versions)
        img = os.path.join(uni_transform_feature_root, f'{slide_name}/{x}_{y}_{random_version}.npy')

    return img

class NoisyTASDataset(Dataset):
    def __init__(
            self,
            df, 
            transform,
            mode = 'train',
            split_col = 'train_test_split', fold = None, random_seed = 0, 
            uni_feature_path = None, uni_transform_feat_path = None,
            num_versions = 50,
            percent_orig = 0.1,
            uni_model = None,
            rank = 0,
            label_col = 'label'
        ):
        self.transform = transform
        self.random_seed = random_seed
        self.mode = mode
        self.split_col = split_col
        self.fold = fold
        self.uni_feature_path = uni_feature_path
        self.uni_transform_feat_path = uni_transform_feat_path
        self.num_versions = num_versions
        self.percent_orig = percent_orig
        self.uni_model = uni_model
        self.rank = rank
        self.label_col = label_col
    

        self.df = self._filter_dataframe(df, split_col, mode, fold)
        logging.info(f'Mode: {self.mode}, filtered dataframe length: {len(self.df)}')

        if mode in ['labeled', 'unlabeled']:
            self._apply_labeled_unlabeled_filtering(mode)

    def _apply_labeled_unlabeled_filtering(self, mode):
        assert self.pred is not None and self.probability is not None, 'pred and prob must be provided'
        mask = self.pred.astype('bool') if mode == 'labeled' else ~self.pred.astype('bool')

        if len(mask) != len(self.df):
            logging.warning(f'Length mismatch between mask and dataframe, {mask.shape}, {self.df.shape}')

        self.df = self.df[mask].reset_index(drop = True)
        self.probability = self.probability[mask]

    def _filter_dataframe(self, df, split_col, mode, fold):
        if split_col == 'train_test_split':

            if mode == 'warmup' or mode == 'eval_train':
                return df[df[split_col] == 'train'].reset_index(drop = True)
            
            elif mode == 'val':
                return df[df[split_col] == 'val'].reset_index(drop = True)
            
            elif mode == 'test':
                return df[df[split_col] == 'test'].reset_index(drop = True)
            
            elif mode in ['labeled', 'unlabeled']:
                return df[df[split_col] == 'train'].reset_index(drop = True)
            
        else:
            raise ValueError('invalid split col selection')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        curr_row = self.df.loc[index]
        x, y = curr_row['x'], curr_row['y']
        slide_name = curr_row['slide_name']

        img_path = os.path.join(curr_row.patch_root, f'{x}_{y}.jpeg')
        if self.mode in ['val', 'test']:
            try:
                image = Image.open(img_path).convert('RGB')
            except FileNotFoundError:
                print(f'File not found: {img_path}')
                return None
            except Exception as e:
                print(f'Error loading img: {img_path}, {str(e)}')
                return None

            img = self.transform(image)
            label = self.df.loc[index, self.label_col]
            label_binary = self.df.loc[index, 'label']
            return img, label, label_binary

        else:
            img = get_random_transform(
                self.uni_feature_path, 
                self.uni_transform_feat_path, 
                x, y, slide_name, self.num_versions, self.percent_orig)
            try:
                img = np.load(img)
            except Exception as e:
                #print(f'Error loading img: {img}, {str(e)}')
                #return None
                print(f'Issue with uni embed: {x}, {y}, {slide_name}')
                img_path = os.path.join(curr_row.patch_root, f'{x}_{y}.jpeg')
                image = Image.open(img_path).convert('RGB')
                img = self.transform(image)
                device = torch.device(f'cuda:{self.rank}')
                img = img.to(device)
                embeddings = self.uni_model(img)
                img = embeddings
            
            if img is None:
                return None
            if len(img.shape) != 1:
                return None

            if self.mode == 'warmup':
                label = self.df.loc[index, self.label_col]
                label_binary = self.df.loc[index, 'label']
                return img, label, label_binary

            
class NoisyDataloader():
    def __init__(
            self,
            df,
            batch_size,
            num_workers,
            args,
            fold = None,
            world_size = 1,
            rank = 0,
            distributed = False,
            uni_model = None,
            prefetch = True,
            label_col = 'label'):
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df = df
        self.args = args
        self.split_col = args.split_col

        self.fold = fold
        self.world_size = world_size
        self.rank = rank
        self.distributed = distributed

        self.uni_model = uni_model
        self.prefetch = prefetch

        self.label_col = label_col

        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            ])   

    def run(self, mode):
        if mode == 'train':
            labeled_dataset = NoisyTASDataset(
                self.df, 
                self.transform_train,
                mode = 'warmup',
                split_col = self.split_col, 
                random_seed = self.args.seed, 
                uni_feature_path = self.args.uni_feature_path, 
                uni_transform_feat_path = self.args.uni_transform_feat_path,
                num_versions = self.args.num_versions,
                percent_orig = self.args.percent_orig,
                uni_model=self.uni_model,
                rank = self.rank,
                label_col = self.label_col
            )
            labeled_loader = DataLoader(
                labeled_dataset,
                batch_size = self.batch_size,
                num_workers = self.num_workers,
                drop_last = True,
                shuffle = True,
                pin_memory = True,
                collate_fn = custom_collate_fn
            )
            if self.prefetch:
                return PrefetchLoader(labeled_loader)
            return labeled_loader

        elif mode in ['val', 'test']:
            dataset = NoisyTASDataset(
                self.df,
                self.transform_test,
                mode = mode,
                split_col = self.split_col,
                fold = self.fold,
                uni_model=self.uni_model,
                rank = self.rank,
                label_col = self.label_col
            )
            
            loader = DataLoader(
                dataset,
                batch_size = self.batch_size,
                shuffle = False,
                num_workers = self.num_workers,
                pin_memory = True,
                collate_fn = custom_collate_fn
            )
            if self.prefetch:
                return PrefetchLoader(loader)
            return loader

