import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from sklearn.metrics import (
    roc_curve, 
    roc_auc_score, 
    recall_score, 
    precision_score, 
    accuracy_score, 
    confusion_matrix,
    average_precision_score
    )

def run_threshold_metrics(model_outputs, arr_labels, one_hot = False):
    if torch.is_tensor(model_outputs):
        model_outputs = model_outputs.cpu().numpy()

    if one_hot == True:
        arr_scores = model_outputs[:, 1]
    else:
        arr_scores = model_outputs

    false_pos_rate, true_pos_rate, proba = roc_curve(arr_labels, arr_scores)
    auc = roc_auc_score(arr_labels, arr_scores)

    optimal_proba_cutoff = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
    arr_results = np.array(arr_scores > optimal_proba_cutoff).astype(np.uint8)

    recall = recall_score(arr_labels, arr_results)
    precision = precision_score(arr_labels, arr_results)
    acc = accuracy_score(arr_labels, arr_results)

    cm1 = confusion_matrix(arr_labels, arr_results)

    sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    avg_prc = average_precision_score(arr_labels, arr_scores)

    return (auc, acc, recall, precision, sensitivity, specificity, cm1, optimal_proba_cutoff, avg_prc)


def init_net(net):
    net.apply(init_weights)

def init_weights(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.kaiming_normal_(m.weight.data, mode = 'fan_out', nonlinearity = 'relu', a = 0)
    elif 'Linear' in classname:
        init.kaiming_normal_(m.weight.data, a = 0)
        init.constant_(m.bias.data, 0)
    elif 'BatchNorm' in classname:
        init.normal_(m.weight.data, 1.0, 2.0)
        init.constant_(m.bias.data, 0.0)


class AverageMeter(object):
    def __init__(self, name, fmt = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def get_avg(self):
        return self.avg


def save_model(net, optimizer, epoch, path):
    torch.save({
        'epoch': epoch, 
        'model_stae_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

    print(f'Model saved! {epoch}, path {path}')

def extract_uni_batch(images, uni_model, batch_size):
    device = torch.device(f'cuda:0')
    remaining = images.shape[0]
    if remaining != batch_size:
        padding = torch.zeros((batch_size - remaining,) + images.shape[1:]).type(images.type())
        images = torch.vstack([images, padding])
    images = images.to(device) #king = True)#.cuda()
    with torch.no_grad():
        embeddings = uni_model(images)
        if remaining != batch_size:
            embeddings = embeddings[:remaining]

    return embeddings

def linear_rampup(current, warm_up, rampup_length = 16):
    current = np.clip((current-warm_up)/rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)