import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from model import *

from dataloader_simple import *
from utils import run_threshold_metrics
import torch.utils.tensorboard as tensorboard
from logger import *
import pickle as pkl
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

parser = argparse.ArgumentParser('One hot encoding training - divide mix TAS')
parser.add_argument(
    '--batch_size', default = 64, type = int
)
parser.add_argument(
    '--lr', '--learning_rate', type = float, default = 1e-4
)
parser.add_argument(
    '--alpha', type = float, default = 0.5, help = 'parameter for beta'
)
parser.add_argument(
    '--lambda_u', type = float, default = 0.25, help = 'weight for unsupervised loss'
)
parser.add_argument(
    '--T', type = float, default = 0.8, help = 'sharpening temp'
)
parser.add_argument(
    '--p_threshold', type = float, default = 0.5, help = 'clean prob threshold'
)
parser.add_argument(
    '--num_epochs', type = int, default = 50
)
parser.add_argument(
    '--num_warmup_epochs', type = int, default = 2
)
parser.add_argument(
    '--id', type = str, default = 'one_hot_tas'
)
parser.add_argument(
    '--data_path', type = str, desc = 'Path to csv for monte carlo dropout - should have x, y, path to features'
)
parser.add_argument(
    '--seed', type = int, default = 123
)
parser.add_argument(
    '--num_class', type = int, default = 1
)
parser.add_argument(
    '--num_batches', type = int, default = 5000
)
parser.add_argument(
    '--uni_checkpoint', type = str, desc = 'Path to uni model'
)
parser.add_argument(
    '--in_channels', type = int, default = 1024)
parser.add_argument(
    '--dropout_prob', type = float, default = 0.25)
parser.add_argument(
    '--initalize_weights', type = bool, default = True)
parser.add_argument(
    '--save_root', type = str, desc = 'Where to save predictions'
)
parser.add_argument(
    '--exp_name', type = str, default = ''
)
parser.add_argument(
    '--split_col', type = str, default = 'train_test_split'
)
parser.add_argument(
    '--fold', default = None
)
parser.add_argument(
    '--debug', type = bool, default = False
)
parser.add_argument(
    '--uni_feature_path', type = str, default = 'Uni feature root'
)
parser.add_argument(
    '--uni_transform_feat_path', type = str, default = 'Uni transforms presaved'
)
parser.add_argument(
    '--num_versions', type = int, default = 49 
)
parser.add_argument(
    '--percent_orig', type = float, default = 0.1
)
parser.add_argument(
    '--prefetch', type = bool, default = False
)
parser.add_argument(
    '--conf_alpha', type = float, default = 0.5
)
parser.add_argument(
    '--label_col', type = str, default = 'label'
)
parser.add_argument(
    '--l1', type = int, default = 128
)
parser.add_argument(
    '--l2', type = int, default = 64
)
parser.add_argument(
    '--best_model_checkpoint', type = str, default = ''
)
parser.add_argument(
    '--forward_passes', type = int, default = 100
)

args = parser.parse_args()

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return []
    return default_collate(batch)

class TestDataset(Dataset):
    def __init__(self,
        df,
        uni_feature_path,
        rank = 0
        ):
        self.df = df
        self.uni_feature_path = uni_feature_path
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        curr_row = self.df.loc[index]
        x, y = curr_row['x'], curr_row['y']
        slide_name = curr_row['slide_name']
        feat_path = os.path.join(self.uni_feature_path, slide_name, f'{x}_{y}.npy')
        try:
            feats = np.load(feat_path)
        except:
            return None, None
        label = curr_row['label']
        return feats, label

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def extract_uni_batch(images, uni_model, batch_size):
    device = torch.device(f'cuda:0')
    remaining = images.shape[0]
    if remaining != batch_size:
        padding = torch.zeros((batch_size - remaining,) + images.shape[1:]).type(images.type())
        images = torch.vstack([images, padding])
    images = images.to(device)
    with torch.no_grad():
        embeddings = uni_model(images)
        if remaining != batch_size:
            embeddings = embeddings[:remaining]

    return embeddings

class NegEntropy_Binary(object):
    def __call__(self, outputs):
        probs = torch.sigmoid(outputs)
        return torch.mean(probs * torch.log(probs) + (1-probs) * torch.log(1 - probs))

def create_model(args):
    model = MLPClassifier(
                    in_channels = args.in_channels, 
                    layers = [args.l1, args.l2], 
                    out_channels = 1,
                    dropout_prob = args.dropout_prob,
                    initalize_weight = args.initalize_weights)
    
    return model

def create_uni_model(args, device):
    uni_model = timm.create_model(
        "vit_large_patch16_224", 
        img_size = 224, 
        patch_size = 16, 
        init_values = 1e-5, 
        num_classes = 0, 
        dynamic_img_size = True
        )

    uni_model.load_state_dict(torch.load(args.uni_checkpoint), strict = True)
    uni_model = uni_model.to(device)

    return uni_model

def test(model, test_loader, args):
    device = torch.device(f'cuda:0')

    correct = 0
    total = 0

    n_samples = len(test_loader.dataset)
    dropout_predictions = np.empty((0, n_samples, 1))


    for i in tqdm(range(args.forward_passes)):
        predictions = np.empty((0, 1))
        model.eval()
        enable_dropout(model)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                
                inputs = inputs.to(device)
                outputs = model(inputs)

                probs = torch.sigmoid(outputs)
                probs = probs.cpu().numpy()

                for pred in probs:
                    predictions = np.vstack((predictions, pred))
            
        dropout_predictions = np.vstack((
            dropout_predictions, predictions[np.newaxis, :, :]
        ))


    return dropout_predictions

def main(rank, world_size):
    device = torch.device(f'cuda:{rank}')
    save_folder_results = os.path.join(args.save_root, args.exp_name,'holdout_test_ucla')
    os.makedirs(save_folder_results, exist_ok = True)

    net = create_model(args)
    #uni_model = create_uni_model(args, device)

    checkpoint = torch.load(args.best_model_checkpoint)
    print(checkpoint.keys())
    net.load_state_dict(checkpoint)
    net.cuda(device)

    df = pd.read_csv(args.data_path)
    test_df = df[df[args.split_col] == 'test'].reset_index(drop = True)
    test_dataset = TestDataset(
        test_df, args.uni_feature_path
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = 2,
        pin_memory = True,
        collate_fn = custom_collate_fn
    )
    dropout_predictions = test(net, test_loader, args)

    np.save(os.path.join(save_folder_results, f'test_results_with_mcd.npy'), dropout_predictions)

if __name__ == "__main__":
    world_size = 1
    main(rank = 0, world_size = 1)
