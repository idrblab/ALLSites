"""
Helper functions for training and data processing.
"""

import torch
import numpy as np
import random
from typing import Tuple, List


def init_seeds(seed: int = 42):
    """Initialize random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def todevice(local, protein, label, local_num, protein_num, device):
    """Move data tensors to specified device."""
    local = torch.from_numpy(local).float().to(device)
    protein = torch.from_numpy(protein).float().to(device)
    label = torch.from_numpy(label).long().to(device)
    
    return local, protein, label, local_num, protein_num


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, filename):
    """Save training checkpoint."""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_score', 0)
