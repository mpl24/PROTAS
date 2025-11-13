import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch.optim as optim

class MLPClassifier(nn.Module):
    def __init__(
        self, 
        in_channels, 
        layers = [128, 64], 
        out_channels = 1,
        dropout_prob = 0.25,
        initalize_weight = False):
        super(MLPClassifier, self).__init__()

        self.in_channels = in_channels
        self.layers = layers
        self.out_channels = out_channels
        self.dropout_prob = dropout_prob
        self.initalize_weights = initalize_weight

        self.classifier = nn.Sequential(
            nn.Linear(self.in_channels, self.layers[0]),
            nn.BatchNorm1d(self.layers[0]),
            nn.ReLU(),
            nn.Dropout(p = self.dropout_prob),
            nn.Linear(self.layers[0], self.layers[1]),
            nn.BatchNorm1d(self.layers[1]),
            nn.ReLU(),
            nn.Dropout(p = self.dropout_prob),
            nn.Linear(self.layers[1], self.out_channels)
        )
        if self.initalize_weights is True:
            for m in self.children():
                init_net(m)

    def forward(self, x):
        x = self.classifier(x)
        return x
