# PROTAS Model
## Overview

PROTAS uses a 2-layer MLP classifier trained on UNI foundation model embeddings (1024-dim) to predict reactive stroma in prostate tissue.

## Model Architecture


```python
class MLPClassifier(nn.Module):
    """
    Input: 1024-dim UNI embeddings
    Hidden 1: 128 neurons (BatchNorm + ReLU + Dropout)
    Hidden 2: 64 neurons (BatchNorm + ReLU + Dropout)
    Output: 1 (binary classification)
    """
```

**Hyperparameters:**
- Input dimension: 1024 (UNI embeddings)
- Hidden layers: [128, 64]
- Dropout: 0.25
- Activation: ReLU
- Output: BCEWithLogitsLoss
- Total parameters: ~140K


## Model Checkpoints

**Loading trained model:**
```python
import torch
from model import MLPClassifier

# Load checkpoint
checkpoint = torch.load('best_model.pth')

# Create model
model = MLPClassifier(
    in_channels=1024,
    layers=[128, 64],
    out_channels=1,
    dropout_prob=0.25
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# For Monte Carlo dropout inference
model.train()  # Keeps dropout active
```


## Dependencies

```bash
pip install torch torchvision timm scikit-learn pandas numpy tqdm
```


## Notes

- Model requires pre-computed UNI embeddings (1024-dim)
- Trained on ~1.5 million patches from prostate tissue
- Dropout enabled during inference for uncertainty
- 100 Monte Carlo forward passes recommended