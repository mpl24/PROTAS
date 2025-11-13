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

parser = argparse.ArgumentParser('RP Training')
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
    '--data_path', type = str, desc = 'Path to data info'
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
    '--save_root', type = str, desc = 'Where to save model checkpoints'
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
    '--pre_warmed_up', type = bool, default = True
)
parser.add_argument(
    '--warmup_root', type = str
)
parser.add_argument(
    '--warmup_last_epoch', type = int, default = 4
)
parser.add_argument(
    '--distributed', type = bool, default = False
)
parser.add_argument(
    '--uni_feature_path', type = str, desc = 'Path to uni embeddings'
)
parser.add_argument(
    '--uni_transform_feat_path', type = str, desc = 'Path to uni transform embeddings (for training)'
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

args = parser.parse_args()

torch.cuda.set_device(0)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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

def train(net, optimizer, dataloader, args, uni_model, criterion_warmup, conf_penalty, epoch, meters_train):
    net.train()
    device = torch.device(f'cuda:0')
    all_outputs = []
    all_targets = []
    for batch_idx, (inputs, labels, labels_binary) in enumerate(tqdm(dataloader)):
        inputs, labels = inputs.to(device, non_blocking = True), labels.to(device, non_blocking = True) #cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = outputs.squeeze()
        loss = criterion_warmup(outputs, labels.float())
        penalty = conf_penalty(outputs)
        loss = loss + penalty * args.conf_alpha
        loss.backward()

        optimizer.step()
        all_outputs.append(outputs)
        all_targets.append(labels_binary)

    all_outputs = torch.cat(all_outputs, dim = 0)
    all_targets = torch.cat(all_targets, dim = 0)

    metrics = run_threshold_metrics(all_outputs.detach().cpu().numpy(), all_targets.detach().cpu().numpy(), one_hot = False)
    auc, acc, recall, precision, sensitivity, specificity, cm, _, avg_prc = metrics

    logging.info(f"Warmup Epoch: {epoch}, "
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
    
    return net, optimizer


def val(net, val_loader, k, uni_model, args, metrics_meters_dict, curr_network_ind, best_aucs, epoch):
    net.eval()

    correct = 0
    total = 0

    all_outputs = []
    all_targets = []

    device = torch.device(f'cuda:0')

    with torch.no_grad():
        for batch_idx, (imgs, targets, labels_binary) in enumerate(tqdm(val_loader)):
            inputs = extract_uni_batch(imgs, uni_model, args.batch_size)
            inputs, targets = inputs.to(device, non_blocking = True), targets.to(device, non_blocking = True)#inputs.cuda(), targets.cuda()
            outputs1 = net(inputs)

            probs = torch.sigmoid(outputs1)

            all_outputs.append(probs)
            all_targets.append(labels_binary)

    all_outputs = torch.cat(all_outputs, dim = 0)
    all_targets = torch.cat(all_targets, dim = 0)

    metrics = None

    metrics = run_threshold_metrics(all_outputs.cpu().numpy(), all_targets.cpu().numpy(), one_hot = False)
    auc, acc, recall, precision, sensitivity, specificity, cm, _, avg_prc = metrics
    
    if auc > best_aucs[0]:
        best_aucs[0] = auc
        print(f'Saving best net: {k}')
        save_point = os.path.join(args.save_root, args.exp_name,'checkpoints', f'{args.id}_{k}_best.pth.tar')
        torch.save(net.state_dict(), save_point)

    batch_len = len(all_outputs)
    metrics_meters_dict['auc'].update(auc, batch_len)
    metrics_meters_dict['auprc'].update(avg_prc, batch_len)
    metrics_meters_dict['sensitivity'].update(sensitivity, batch_len)
    metrics_meters_dict['specificity'].update(specificity, batch_len)
    metrics_meters_dict['acc'].update(acc, batch_len)

    logging.info(f"Val Epoch: {epoch}, network: {curr_network_ind}, "
        f"# samples: {len(val_loader)*args.batch_size}, AUC: {metrics_meters_dict['auc'].avg:.3f}, "
        f"AUPRC: {metrics_meters_dict['auprc'].avg:.3f}, SEN: {metrics_meters_dict['sensitivity'].avg:.3f}, "
        f" SPEC: {metrics_meters_dict['specificity'].avg:.3f}, ACC: {metrics_meters_dict['acc'].avg:.3f}")

    del all_outputs, all_targets
    torch.cuda.empty_cache()

    return metrics

def main(rank, world_size):
    device = torch.device(f'cuda:{rank}')

    save_folder_checkpoints = os.path.join(args.save_root, args.exp_name, f'checkpoints')
    save_folder_logger = os.path.join(args.save_root, args.exp_name, f'logger')
    save_folder_writer = os.path.join(args.save_root, args.exp_name, f'writer')

    os.makedirs(save_folder_checkpoints, exist_ok=True)
    os.makedirs(save_folder_logger, exist_ok=True)
    os.makedirs(save_folder_writer, exist_ok=True)

    print('Setting up logging...')
    log_filename = 'out.log'
    args.log_path = os.path.join(save_folder_logger , log_filename)
    args.log_level = logging.INFO
    setup_logging(args.log_path, args.log_level)

    loss_meter_1 = AverageMeter('train')

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
        'acc': AverageMeter('val')
    }

    logging.info('Building network')

    net1 = create_model(args).to(device)
    uni_model = create_uni_model(args, device)

    cudnn.benchmark = True

    optimizer1 = optim.SGD(net1.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-3)
    scheduler = CosineAnnealingLR(optimizer1, T_max = args.num_epochs+1)

    criterion = nn.BCEWithLogitsLoss()
    conf_penalty = NegEntropy_Binary()

    df = pd.read_csv(args.data_path)
    loader = NoisyDataloader(
        df, batch_size = args.batch_size, 
        num_workers = 4, args = args, 
        fold = args.fold, 
        world_size = 1, rank = 0, distributed = False, uni_model = uni_model, prefetch = args.prefetch)
    best_aucs = [0]

    metrics_dict = {}
    logging.info(f'Len of full dataset: {len(df)}')

    if args.start_from_checkpoint == True:
        check = torch.load(args.checkpoint_path)
        net1.load_state_dict(checkpoint['model_stae_dict'])
        net1.to(device)
    
    for epoch in range(args.start_epoch,args.num_epochs + 1):
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        warmup_loader = loader.run('train')
        logging.info(f'Len dataset : {len(warmup_loader)}')
        logging.info('Train...')
        net1, optimizer1 = train(
            net1, 
            optimizer1, 
            warmup_loader, 
            args, uni_model, 
            criterion, 
            conf_penalty, 
            epoch,
            metric_meters_1_train
            )

        scheduler.step()
        
        del warmup_loader
        torch.cuda.empty_cache()

        val_loader = loader.run('val')
        metrics1 = val(net1, val_loader, 1, uni_model, args, metric_meters_1_val, curr_network_ind = 1, best_aucs = best_aucs, epoch = epoch)

        metrics_dict[epoch] = [metrics1]

        del val_loader
        torch.cuda.empty_cache()

        logging.info(f'Completed epoch: {epoch}')
        save_model(net1, optimizer1, epoch, path = os.path.join(save_folder_checkpoints, f"net1_{epoch}_{metrics1[0]}.pth"))

    with open(os.path.join(save_folder_writer, f'metrics_{args.exp_name}.pkl'), 'wb') as f:
        pkl.dump(metrics_dict, f)

if __name__ == "__main__":
    world_size = 1
    main(rank = 0, world_size = 1)
