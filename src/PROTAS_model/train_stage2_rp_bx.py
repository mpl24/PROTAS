import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.tensorboard as tensorboard
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

import random
import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

from model import *
from dataset import *
from utils import *
from logger import *

parser = argparse.ArgumentParser('Biopsy + RP')
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
    '--data_path', type = str, default = ''
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
    '--uni_checkpoint', type = str, default = ''
)
parser.add_argument(
    '--in_channels', type = int, default = 1024)
parser.add_argument(
    '--dropout_prob', type = float, default = 0.25)
parser.add_argument(
    '--initalize_weights', type = bool, default = True)
parser.add_argument(
    '--save_root', type = str, desc = 'Where to save checkpoints/logs etc'
)
parser.add_argument(
    '--exp_name', type = str, default = 'rp_biopsy_DANN_chunked_lambda_0.75_adam_50+_unbalanced'
)
parser.add_argument(
    '--split_col', type = str, default = None
)
parser.add_argument(
    '--debug', type = bool, default = False
)
parser.add_argument(
    '--distributed', type = bool, default = False
)
parser.add_argument(
    '--rp_uni_feature_path', type = str, default = ''
)
parser.add_argument(
    '--rp_uni_transform_feat_path', type = str, default = None
)
parser.add_argument(
    '--biopsy_uni_feature_path', type = str, default = ''
)
parser.add_argument(
    '--biopsy_uni_transform_feat_path', type = str, default = None
)
parser.add_argument(
    '--num_versions', type = int, default = 49
)
parser.add_argument(
    '--percent_orig', type = float, default = 1.0
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
## options to start from a presaved checkpoint and continue training
parser.add_argument(
    '--start_from_checkpoint', type = bool, default = False
)
parser.add_argument(
    '--start_epoch', type = int, default = 0
)
parser.add_argument(
    '--checkpoint_path', type = str, default = None
)
parser.add_argument(
    '--train_data_path', type = str, desc = 'Train data csv'
)
parser.add_argument(
    '--val_data_path', type = str, desc = 'Val data csv'
)
parser.add_argument(
    '--shard_data_root', type = str, desc = 'Test data csv'
)
parser.add_argument(
    '--lambda_domain', type = float, default = 0.75
)
parser.add_argument(
    '--scheduler', type = bool, default = False
)
parser.add_argument(
    '--neg_entropy', type = bool, default = False
)
parser.add_argument(
    '--label_smoothing', type = bool, default = False
)

args = parser.parse_args()

torch.cuda.set_device(0)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GRL(nn.Module):
    def __init__(self, alpha = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class NegEntropy_Binary(object):
    def __call__(self, outputs):
        probs = torch.sigmoid(outputs)
        return torch.mean(probs * torch.log(probs) + (1-probs) * torch.log(1 - probs))


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim = 1024, hidden_dim = 512, hidden_dim_2 = 128, dropout_prob = 0.25, initalize_weights = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout_prob),
            nn.Linear(hidden_dim, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p = dropout_prob)
        )
        if initalize_weights is True:
            for m in self.children():
                init_net(m)

    def forward(self, x):
        return self.net(x)

class LabelPredictor(nn.Module):
    def __init__(self, hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier(nn.Module):
    def __init__(self, hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def create_model(args):
    feature_extractor = FeatureExtractor(
        input_dim=args.in_channels, 
        hidden_dim=args.l1,  
        hidden_dim_2=args.l2,  
        dropout_prob=args.dropout_prob,
        initalize_weights=args.initalize_weights
    )
    
    label_predictor = LabelPredictor(hidden_dim=args.l2)
    domain_classifier = DomainClassifier(hidden_dim=args.l2)
    
    class DANNModel(nn.Module):
        def __init__(self, feature_extractor, label_predictor, domain_classifier):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.label_predictor = label_predictor
            self.domain_classifier = domain_classifier
            self.grl = GRL(alpha=1.0) 
            
        def forward(self, x, compute_domain=True, alpha = None):
            features = self.feature_extractor(x)
            class_output = self.label_predictor(features)

            if alpha is not None:
                self.grl.alpha = alpha
            
            if compute_domain:
                reversed_features = self.grl(features)
                domain_output = self.domain_classifier(reversed_features)
                return class_output, domain_output
            return class_output
            
    model = DANNModel(feature_extractor, label_predictor, domain_classifier)
    return model


def train(net, optimizer, train_loader, train_length, args, uni_model, criterion, epoch, meters_train, conf_penalty = None):
    net.train()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    all_outputs = []
    all_targets = []

    pbar = tqdm(train_loader, total=train_length//args.batch_size, desc='Training Loop')
    domain_criterion = nn.BCEWithLogitsLoss()

    p = float(epoch) / args.num_epochs
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    try:
        for batch_idx, (batch) in enumerate(pbar):
            batch_features = batch[:, :1024].to(device, non_blocking=True)

            if args.label_smoothing:
                batch_labels = batch[:, 1026:1027].to(device, non_blocking=True)
                binary_labels = batch[:, 1024:1025].to(device, non_blocking=True)
            else:
                batch_labels = batch[:, 1024:1025].to(device, non_blocking=True)#.unsqueeze(1)
            domain_labels = batch[:, 1025:1026].to(device, non_blocking=True)#.unsqueeze(1)
            optimizer.zero_grad()

            class_outputs, domain_outputs = net(batch_features, compute_domain=True)
            
            class_loss = criterion(class_outputs, batch_labels.float())
            domain_loss = domain_criterion(domain_outputs, domain_labels.float())
            
            if conf_penalty is not None:
                penalty = conf_penalty(class_outputs)
                total_loss = class_loss - args.lambda_domain * domain_loss + penalty * args.conf_alpha
            else:
                total_loss = class_loss - args.lambda_domain * domain_loss

            if torch.isnan(total_loss):
                logging.warning(f"NaN loss detected at batch {batch_idx}")
                continue
                
            total_loss.backward()

            optimizer.step()
            all_outputs.append(class_outputs)
            if args.label_smoothing:
                all_targets.append(binary_labels.detach().cpu())
            
            else:
                all_targets.append(batch_labels.detach().cpu())

            if batch_idx % 500 == 0 and batch_idx > 0:
                outputs_cat = torch.cat(all_outputs, dim=0)
                targets_cat = torch.cat(all_targets, dim=0)
                metrics = run_threshold_metrics(outputs_cat.detach().cpu().numpy(),
                                                targets_cat.detach().cpu().numpy(),
                                                one_hot=False)
                auc, _, _, _, _, _, _, _, avg_prc = metrics
                pbar.set_postfix(auc=f"{auc:.3f}", auprc=f"{avg_prc:.3f}")

    except Exception as e:
        logging.error(f"Error in training loop: {str(e)}")
        raise

    all_outputs = torch.cat(all_outputs, dim = 0)
    all_targets = torch.cat(all_targets, dim = 0)

    metrics = run_threshold_metrics(all_outputs.detach().cpu().numpy(), all_targets.detach().cpu().numpy(), one_hot = False)
    auc, acc, recall, precision, sensitivity, specificity, cm, _, avg_prc = metrics

    logging.info(f"Train Epoch: {epoch}, "
        f"AUC: {auc:.3f}, "
        f"AUPRC: {avg_prc:.3f}, SEN: {sensitivity:.3f}, "
        f" SPEC: {specificity:.3f}, ACC: {acc:.3f}")

    batch_len = len(all_outputs.detach().cpu().numpy())
    meters_train['auc'].update(auc, batch_len)
    meters_train['auprc'].update(avg_prc, batch_len)
    meters_train['sensitivity'].update(sensitivity, batch_len)
    meters_train['specificity'].update(specificity, batch_len)
    meters_train['acc'].update(acc, batch_len)

    del all_outputs, all_targets
    torch.cuda.empty_cache()
    
    return net, optimizer, metrics


def val(net, val_loader, val_length, uni_model, args, metrics_meters_dict, curr_network_ind, best_aucs, epoch):
    net.eval()
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    all_outputs = []
    all_targets = []
    all_outputs_rp = []
    all_targets_rp = []
    all_outputs_biopsy = []
    all_targets_biopsy = []

    try:
        with torch.no_grad():
            for batch_idx, (batch) in enumerate(tqdm(val_loader, desc='Validation', total=val_length//args.batch_size)):
                batch_features = batch[:, :1024].to(device, non_blocking=True)
                batch_labels = batch[:, 1024:1025]
                domain_labels = batch[:, 1025:1026]
            
                class_outputs = net(batch_features, compute_domain=False)
                probs = torch.sigmoid(class_outputs)

                rp_ind = np.where(domain_labels == 1)[0]
                bx_ind = np.where(domain_labels == 0)[0]

                probs_rp = probs[rp_ind]
                labels_rp = batch_labels[rp_ind]

                probs_bx = probs[bx_ind]
                labels_bx = batch_labels[bx_ind]

                if probs.dim() > 0:
                    all_outputs.append(probs)
                if batch_labels.squeeze().dim() > 0:
                    all_targets.append(batch_labels.squeeze())
                if probs_rp.dim() > 0:
                    all_outputs_rp.append(probs_rp)
                if labels_rp.squeeze().dim() > 0:
                    all_targets_rp.append(labels_rp.squeeze())
                if probs_bx.dim() > 0:
                    all_outputs_biopsy.append(probs_bx)
                if labels_bx.squeeze().dim() > 0:
                    all_targets_biopsy.append(labels_bx.squeeze())

    except Exception as e:
        logging.error(f"Error in validation loop: {str(e)}")
        raise

    if all_outputs:
        all_outputs = torch.cat(all_outputs, dim=0)
    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)
    if all_outputs_rp:
        all_outputs_rp = torch.cat(all_outputs_rp, dim=0)
    if all_targets_rp:
        all_targets_rp = torch.cat(all_targets_rp, dim=0)
    if all_outputs_biopsy:
        all_outputs_biopsy = torch.cat(all_outputs_biopsy, dim=0)
    if all_targets_biopsy:
        all_targets_biopsy = torch.cat(all_targets_biopsy, dim=0)

    if all_outputs.size(0) > 0 and all_targets.size(0) > 0:
        metrics = run_threshold_metrics(all_outputs.cpu().numpy(), all_targets.cpu().numpy(), one_hot=False)
        auc, acc, recall, precision, sensitivity, specificity, cm, _, avg_prc = metrics
    else:
        logging.warning("No valid outputs or targets for overall metrics calculation.")
        return None, None, None

    if all_outputs_rp.size(0) > 0 and all_targets_rp.size(0) > 0:
        if all_outputs_rp.size(0) == all_targets_rp.size(0):
            metrics_rp = run_threshold_metrics(all_outputs_rp.cpu().numpy(), all_targets_rp.cpu().numpy(), one_hot=False)
        else:
            logging.warning("Inconsistent number of samples for RP metrics calculation.")
            metrics_rp = None
    else:
        logging.warning("No valid outputs or targets for RP metrics calculation.")
        metrics_rp = None

    if all_outputs_biopsy.size(0) > 0 and all_targets_biopsy.size(0) > 0:
        if all_outputs_biopsy.size(0) == all_targets_biopsy.size(0):
            metrics_biopsy = run_threshold_metrics(all_outputs_biopsy.cpu().numpy(), all_targets_biopsy.cpu().numpy(), one_hot=False)
        else:
            logging.warning("Inconsistent number of samples for biopsy metrics calculation.")
            metrics_biopsy = None
    else:
        logging.warning("No valid outputs or targets for biopsy metrics calculation.")
        metrics_biopsy = None

    if auc > best_aucs[0]:
        best_aucs[0] = auc
        try:
            save_point = os.path.join(args.save_root, args.exp_name, 'checkpoints', f'{args.id}_best_{epoch}.pth')
            torch.save(net.state_dict(), save_point)
            logging.info(f'Saved best model to {save_point}')
        except Exception as e:
            logging.error(f"Error saving best model: {str(e)}")

    if metrics:
        batch_len = len(all_outputs)
        metrics_meters_dict['auc'].update(auc, batch_len)
        metrics_meters_dict['auprc'].update(avg_prc, batch_len)
        metrics_meters_dict['sensitivity'].update(sensitivity, batch_len)
        metrics_meters_dict['specificity'].update(specificity, batch_len)
        metrics_meters_dict['acc'].update(acc, batch_len)

    if metrics_rp:
        batch_len = len(all_outputs_rp)
        metrics_meters_dict['auc_rp'].update(metrics_rp[0], batch_len)
        metrics_meters_dict['auprc_rp'].update(metrics_rp[8], batch_len)
        metrics_meters_dict['sensitivity_rp'].update(metrics_rp[4], batch_len)
        metrics_meters_dict['specificity_rp'].update(metrics_rp[5], batch_len)
        metrics_meters_dict['acc_rp'].update(metrics_rp[1], batch_len)

    if metrics_biopsy:
        batch_len = len(all_outputs_biopsy)
        metrics_meters_dict['auc_biopsy'].update(metrics_biopsy[0], batch_len)
        metrics_meters_dict['auprc_biopsy'].update(metrics_biopsy[8], batch_len)
        metrics_meters_dict['sensitivity_biopsy'].update(metrics_biopsy[4], batch_len)
        metrics_meters_dict['specificity_biopsy'].update(metrics_biopsy[5], batch_len)
        metrics_meters_dict['acc_biopsy'].update(metrics_biopsy[1], batch_len)

    logging.info(f"Val Epoch: {epoch}, network: {curr_network_ind}, "
        f"# samples: {len(all_outputs)}, AUC: {metrics_meters_dict['auc'].avg:.3f}, "
        f"AUPRC: {metrics_meters_dict['auprc'].avg:.3f}, SEN: {metrics_meters_dict['sensitivity'].avg:.3f}, "
        f" SPEC: {metrics_meters_dict['specificity'].avg:.3f}, ACC: {metrics_meters_dict['acc'].avg:.3f}"
        f" RP METRICS: auc - {metrics_meters_dict['auc_rp'].avg:.3f}, auprc - {metrics_meters_dict['auprc_rp'].avg:.3f}"
        f" BIOPSY METRICS: auc - {metrics_meters_dict['auc_biopsy'].avg:.3f}, auprc - {metrics_meters_dict['auprc_biopsy'].avg:.3f}"
        )

    del all_outputs, all_targets, all_outputs_rp, all_targets_rp, all_outputs_biopsy, all_targets_biopsy
    torch.cuda.empty_cache()

    return metrics, metrics_rp, metrics_biopsy

def main(rank, world_size):
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    save_folder_checkpoints = os.path.join(args.save_root, args.exp_name, 'checkpoints')
    save_folder_logger = os.path.join(args.save_root, args.exp_name, 'logger')
    save_folder_writer = os.path.join(args.save_root, args.exp_name, 'writer')

    os.makedirs(save_folder_checkpoints, exist_ok=True)
    os.makedirs(save_folder_logger, exist_ok=True)
    os.makedirs(save_folder_writer, exist_ok=True)

    print('Setting up logging...')
    log_filename = 'out.log'
    args.log_path = os.path.join(save_folder_logger, log_filename)
    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    loss_meter_1 = AverageMeter('train')

    if args.neg_entropy:
        conf_penalty = NegEntropy_Binary()

    else:
        conf_penalty = None

    metric_meters_1_train = {
        'auc': AverageMeter('train'),
        'auprc': AverageMeter('train'),
        'sensitivity': AverageMeter('train'),
        'specificity': AverageMeter('train'),
        'acc': AverageMeter('train')
    }

    metric_meters_1_val = {
        'auc': AverageMeter('val'),
        'auprc': AverageMeter('val'),
        'sensitivity': AverageMeter('val'),
        'specificity': AverageMeter('val'),
        'acc': AverageMeter('val'),
        'auc_rp': AverageMeter('val'),
        'auprc_rp': AverageMeter('val'),
        'sensitivity_rp': AverageMeter('val'),
        'specificity_rp': AverageMeter('val'),
        'acc_rp': AverageMeter('val'),
        'auc_biopsy': AverageMeter('val'),
        'auprc_biopsy': AverageMeter('val'),
        'sensitivity_biopsy': AverageMeter('val'),
        'specificity_biopsy': AverageMeter('val'),
        'acc_biopsy': AverageMeter('val'),
    }

    logging.info('Building network')

    try:
        net = create_model(args).to(device)
        uni_model = None

        cudnn.benchmark = True

        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
        if args.scheduler:
            scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs+1)

        criterion = nn.BCEWithLogitsLoss()

        train_df = pd.read_csv(args.train_data_path, index_col=0)
        train_length = len(train_df)
        del train_df
        val_df = pd.read_csv(args.val_data_path, index_col=0)
        val_length = len(val_df)
        del val_df

        train_shard_files = sorted(glob.glob(os.path.join(args.shard_data_root,'train/combined*.npy')))
        val_shard_files = sorted(glob.glob(os.path.join(args.shard_data_root,'val/combined*.npy')))

        val_dataset = ChunkedShardedIterableDataset(
                val_shard_files, shuffle_shards = False
            )
        val_loader = DataLoader(
                val_dataset,
                batch_size = args.batch_size,
                num_workers = 6,
                pin_memory = True,
                persistent_workers = True
            )

        best_aucs = [0]
        metrics_dict = {}
        
        if args.start_from_checkpoint:
            try:
                checkpoint = torch.load(args.checkpoint_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.to(device)
                logging.info(f"Loaded checkpoint from {args.checkpoint_path}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")
                raise
        
        for epoch in range(args.start_epoch, args.num_epochs + 1):
            torch.manual_seed(epoch)
            torch.cuda.manual_seed_all(epoch)

            train_dataset = ChunkedShardedIterableDataset(
                train_shard_files, shuffle_shards = True
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size = args.batch_size,
                num_workers = 6,
                pin_memory = True,
                persistent_workers = True
            )
            
            logging.info(f'Len train dataset: {train_length}, len val dataset: {val_length}')
            logging.info('Train...')
            
            net, optimizer, train_metrics = train(
                net, optimizer, train_loader, train_length, 
                args, uni_model, criterion, epoch,
                metric_meters_1_train,
                conf_penalty=conf_penalty
            )

            if args.scheduler:
                scheduler.step()
            
            torch.cuda.empty_cache()
            metrics, metrics_rp, metrics_biopsy = val(
                net, val_loader, val_length,
                uni_model, args, metric_meters_1_val, 
                curr_network_ind=1, best_aucs=best_aucs, 
                epoch=epoch
            )

            metrics_dict[epoch] = {
                'val_all': metrics, 
                'val_rp': metrics_rp, 
                'val_biopsy': metrics_biopsy,
                'train_metrics': train_metrics
            }

            torch.cuda.empty_cache()

            logging.info(f'Completed epoch: {epoch}')
            
            try:
                save_model(
                    net, optimizer, epoch, 
                    path=os.path.join(save_folder_checkpoints, f"model_{epoch}_{metrics[0]:.3f}.pth")
                )
            except Exception as e:
                logging.error(f"Error saving model: {str(e)}")

        with open(os.path.join(save_folder_writer, f'metrics_{args.exp_name}.pkl'), 'wb') as f:
            pkl.dump(metrics_dict, f)
            
    except Exception as e:
        logging.error(f"Error in main training loop: {str(e)}")
        raise


if __name__ == "__main__":
    world_size = 1
    main(rank = 0, world_size = 1)


