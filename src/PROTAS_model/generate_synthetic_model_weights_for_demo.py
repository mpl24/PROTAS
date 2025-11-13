import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from utils import *
from model import MLPClassifier
import argparse

def generate_synthetic_model_weights(
    output_path,
    in_channels = 1024,
    layers = [128, 64],
    out_channels = 1,
    dropout_prob = 0.25,
    seed = 24
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPClassifier(
        in_channels = in_channels,
        layers = layers,
        out_channels = out_channels, 
        dropout_prob = dropout_prob,
        initalize_weight = True
    )

    n_params = sum(p.numel() for p in model.parameters())
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'in_channels': in_channels,
            'layers': layers
        },
        'seed': seed,
    }, output_path)


def main(args):
    model = generate_synthetic_model_weights(
        output_path = args.output_path, seed = 24
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output_path', type = str
    )
    args = parser.parse_args()
    main(args)