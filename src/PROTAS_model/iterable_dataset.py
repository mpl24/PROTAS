
import torch
import multiprocessing
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
import random
import pickle
import glob
import os

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChunkedShardedIterableDataset(IterableDataset):
    def __init__(self, shard_paths, shuffle_shards = True):
        super().__init__()
        self.shard_paths = shard_paths
        self.shuffle_shards = shuffle_shards


    def _iterator_single_worker(self):
        shard_paths_local = list(self.shard_paths)
        if self.shuffle_shards:
            random.shuffle(shard_paths_local)
        
        for shard_path in shard_paths_local:
            shard_data = np.load(shard_path)
            np.random.shuffle(shard_data)
            for row in shard_data:
                yield torch.from_numpy(row).float()

    def _iterator_multi_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self._iterator_single_worker()
        
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        shard_paths_local = list(self.shard_paths)
        if self.shuffle_shards and worker_id == 0:
            random.shuffle(shard_paths_local)
        
        shard_paths_local = shard_paths_local[worker_id::num_workers]

        for shard_path in shard_paths_local:
            shard_data = np.load(shard_path)
            np.random.shuffle(shard_data)
            for row in shard_data:
                yield torch.from_numpy(row).float()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self._iterator_single_worker()
        else:
            return self._iterator_multi_worker()


class ChunkedShardedBalancedIterableDataset(IterableDataset):
    def __init__(self, shard_paths, shuffle_shards=True, balance_ratio=1.0, max_samples_per_shard=None):
       
        super().__init__()
        self.shard_paths = shard_paths
        self.shuffle_shards = shuffle_shards
        self.balance_ratio = balance_ratio
        self.max_samples_per_shard = max_samples_per_shard

    def _load_and_balance_shard(self, shard_path):
        shard_data = np.load(shard_path)

        if self.max_samples_per_shard is not None and shard_data.shape[0] > self.max_samples_per_shard:
            shard_data = shard_data[np.random.choice(shard_data.shape[0], self.max_samples_per_shard, replace=False)]

        positives = [row for row in shard_data if row[-2] == 1]
        negatives = [row for row in shard_data if row[-2] == 0]

        n_pos = len(positives)
        n_neg = int(n_pos / self.balance_ratio) if self.balance_ratio > 0 else len(negatives)

        n_neg = min(n_neg, len(negatives))

        if n_pos == 0 or n_neg == 0:
            return [] 

        sampled_positives = random.choices(positives, k=n_pos)  
        sampled_negatives = random.sample(negatives, k=n_neg)   

        balanced_batch = sampled_positives + sampled_negatives
        random.shuffle(balanced_batch)

        return balanced_batch

    def _iterator_single_worker(self):
        shard_paths_local = list(self.shard_paths)
        if self.shuffle_shards:
            random.shuffle(shard_paths_local)
        
        for shard_path in shard_paths_local:
            balanced_batch = self._load_and_balance_shard(shard_path)
            for row in balanced_batch:
                yield torch.from_numpy(row).float()

    def _iterator_multi_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self._iterator_single_worker()
        
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        shard_paths_local = list(self.shard_paths)
        if self.shuffle_shards and worker_id == 0:
            random.shuffle(shard_paths_local)
        
        shard_paths_local = shard_paths_local[worker_id::num_workers]

        for shard_path in shard_paths_local:
            balanced_batch = self._load_and_balance_shard(shard_path)
            for row in balanced_batch:
                yield torch.from_numpy(row).float()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self._iterator_single_worker()
        else:
            return self._iterator_multi_worker()

class ChunkedPatchIterableDataset(IterableDataset):
    def __init__(self, chunk_dir, shuffle=False, transforms=None, use_mixup=False, mixup_alpha=0.2, debug=False, debug_subset_size=10000):
        super().__init__()
        self.chunk_dir = chunk_dir
        self.shuffle = shuffle
        self.transforms = transforms
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.debug = debug
        self.debug_subset_size = debug_subset_size

        self.chunk_files = sorted([
            f for f in os.listdir(chunk_dir)
            if f.endswith('.pt')
        ])

        self.chunk_metadata = []
        self.chunk_sizes = []
        self.total_size = 0

        for pt_file in self.chunk_files:
            chunk_base = pt_file.replace('.pt', '')
            csv_meta = os.path.join(chunk_dir, f'{chunk_base}.csv')
            df = pd.read_csv(csv_meta, index_col=0)
            self.chunk_metadata.append(df)
            self.chunk_sizes.append(len(df))
            self.total_size += len(df)
            
        if self.debug:
            samples_per_chunk = self.debug_subset_size // len(self.chunk_files)
            self.chunk_sizes = [min(size, samples_per_chunk) for size in self.chunk_sizes]
            self.total_size = sum(self.chunk_sizes)
            print(samples_per_chunk, self.total_size)

    def _mixup_data(self, x, y, domain):
        if not self.use_mixup:
            return x, y, domain, None

        batch_size = x.size(0)
        if batch_size < 2:
            return x, y, domain, None

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        mixed_domain = lam * domain + (1 - lam) * domain[index]
        
        return mixed_x, mixed_y, mixed_domain, lam

    def _process_image(self, image):
        
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        if self.transforms:
            image = self.transforms(image)
            
        return image

    def _get_chunk_indices(self, worker_id=None, num_workers=None):
        if worker_id is None or num_workers is None:
            return list(range(len(self.chunk_files)))
        
        
        return list(range(worker_id, len(self.chunk_files), num_workers))

    def _iterator_single_worker(self):
        chunk_indices = self._get_chunk_indices()
        if self.shuffle:
            random.shuffle(chunk_indices)

        current_chunk = None
        current_chunk_idx = None
        
        for chunk_idx in chunk_indices:
            
            if chunk_idx != current_chunk_idx:
                if current_chunk is not None:
                    del current_chunk
                    torch.cuda.empty_cache()
                
                pt_path = os.path.join(self.chunk_dir, self.chunk_files[chunk_idx])
                current_chunk = torch.load(pt_path, map_location='cpu')
                current_chunk_idx = chunk_idx

            
            chunk_size = self.chunk_sizes[chunk_idx]
            indices = list(range(chunk_size))
            if self.shuffle:
                random.shuffle(indices)

            for local_idx in indices:
                image = current_chunk[local_idx]
                image = self._process_image(image)
                
                meta = self.chunk_metadata[chunk_idx].iloc[local_idx]
                
                domain_label_str = meta.domain_label
                if domain_label_str == 'rp':
                    domain_label = 1
                else:
                    domain_label = 0
                
                yield {
                    'image': image,
                    'label': torch.tensor(meta.label, dtype=torch.float).view(-1, 1),
                    'domain': torch.tensor(domain_label, dtype=torch.float).view(-1, 1),  # Changed to float for MixUp
                    'smoothed_label': torch.tensor(meta.smoothed_label, dtype=torch.float).view(-1, 1)
                }

        if current_chunk is not None:
            del current_chunk
            torch.cuda.empty_cache()

    def _iterator_multi_worker(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self._iterator_single_worker()
        
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        
        chunk_indices = self._get_chunk_indices(worker_id, num_workers)
        if self.shuffle and worker_id == 0:
            random.shuffle(chunk_indices)

        current_chunk = None
        current_chunk_idx = None
        
        for chunk_idx in chunk_indices:

            if chunk_idx != current_chunk_idx:
                if current_chunk is not None:
                    del current_chunk
                    torch.cuda.empty_cache()
                
                pt_path = os.path.join(self.chunk_dir, self.chunk_files[chunk_idx])
                current_chunk = torch.load(pt_path, map_location='cpu')
                current_chunk_idx = chunk_idx

            
            chunk_size = self.chunk_sizes[chunk_idx]
            indices = list(range(chunk_size))
            if self.shuffle:
                random.shuffle(indices)

            for local_idx in indices:
                image = current_chunk[local_idx]
                image = self._process_image(image)
                
                meta = self.chunk_metadata[chunk_idx].iloc[local_idx]

                domain_label_str = meta.domain_label
                if domain_label_str == 'rp':
                    domain_label = 1
                else:
                    domain_label = 0
                
                yield {
                    'image': image,
                    'label': torch.tensor(meta.label, dtype=torch.float).view(-1, 1),
                    'domain': torch.tensor(domain_label, dtype=torch.float).view(-1, 1),  # Changed to float for MixUp
                    'smoothed_label': torch.tensor(meta.smoothed_label, dtype=torch.float).view(-1, 1)
                }

        if current_chunk is not None:
            del current_chunk
            torch.cuda.empty_cache()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self._iterator_single_worker()
        else:
            return self._iterator_multi_worker()