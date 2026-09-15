"""
MNIST MLP Assignment (Student Version)

Goal:
Build, train, and evaluate a Multi-Layer Perceptron (MLP) on MNIST.

Rules:
- Use PyTorch only
- Do NOT change function signatures
- Fill in all TODOs
- Keep code modular and readable

You will implement:
1. Model definition
2. Forward pass
3. Loss computation
4. Accuracy calculation
5. Training loop
6. Evaluation loop
7. Prediction
8. Data loading
9. Optimizer creation
10. Checkpoint save/load
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# -------------------------
# 1. Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10):
        super().__init__()

        # TODO:
        # Define a 2-layer MLP:
        # Linear -> ReLU -> Linear
        raise NotImplementedError

    def forward(self, x):
        # TODO:
        # Flatten input and run forward pass
        raise NotImplementedError


# -------------------------
# 2. Loss Function
# -------------------------
def compute_loss(logits, targets):
    # TODO:
    # Use CrossEntropyLoss
    raise NotImplementedError


# -------------------------
# 3. Accuracy
# -------------------------
def compute_accuracy(logits, targets):
    # TODO:
    # Compute classification accuracy
    raise NotImplementedError


# -------------------------
# 4. Optimizer
# -------------------------
def create_optimizer(model, lr=1e-3):
    # TODO:
    # Use Adam optimizer
    raise NotImplementedError


# -------------------------
# 5. Train Step
# -------------------------
def train_step(model, batch, optimizer):
    # TODO:
    # 1. Set model to train
    # 2. Forward pass
    # 3. Compute loss
    # 4. Backward
    # 5. Optimizer step
    raise NotImplementedError


# -------------------------
# 6. Eval Step
# -------------------------
def eval_step(model, batch):
    # TODO:
    # 1. Set model to eval
    # 2. Disable gradients
    # 3. Compute loss and accuracy
    raise NotImplementedError


# -------------------------
# 7. Training Loop
# -------------------------
def train_epoch(model, loader, optimizer):
    # TODO:
    # Loop over batches and average loss
    raise NotImplementedError


# -------------------------
# 8. Evaluation Loop
# -------------------------
def evaluate(model, loader):
    # TODO:
    # Loop over batches and return avg loss and accuracy
    raise NotImplementedError


# -------------------------
# 9. Prediction
# -------------------------
def predict(model, x):
    # TODO:
    # Return predicted class indices
    raise NotImplementedError


# -------------------------
# 10. Data Loaders
# -------------------------
def get_dataloaders(batch_size=64):
    # TODO:
    # Load MNIST with normalization
    raise NotImplementedError


# -------------------------
# 11. Save Model
# -------------------------
def save_checkpoint(model, path):
    # TODO:
    raise NotImplementedError


# -------------------------
# 12. Load Model
# -------------------------
def load_checkpoint(model, path):
    # TODO:
    raise NotImplementedError