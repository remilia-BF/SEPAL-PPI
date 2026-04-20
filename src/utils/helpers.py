"""
Helper utilities for SEPAL-PPI
"""

import torch
import os
import time
import random
import numpy as np
from tqdm import tqdm


def set_random_seed(seed=0):
    """Set random seed for reproducibility across all libraries"""
    # Python random module
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    
    # CUDA random (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    """Get the best available device (CUDA if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_generator(seed=0):
    """Create a torch.Generator with fixed seed for reproducible DataLoader"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id, base_seed=42):
    """Worker initialization function for DataLoader to ensure reproducibility"""
    # 确保每个worker有不同但可预测的种子
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_progress_bar(total, desc="Progress", leave=True, disable=False):
    """Create a tqdm progress bar"""
    return tqdm(total=total, desc=desc, leave=leave, disable=disable, 
                bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')


def update_progress_bar(pbar, n=1):
    """Update progress bar by n steps"""
    if pbar is not None:
        pbar.update(n)


def close_progress_bar(pbar):
    """Close progress bar"""
    if pbar is not None:
        pbar.close()


def print_progress_simple(current, total, desc="Progress", disable=False):
    """Simple progress print without persistent bar"""
    if not disable:
        percent = (current / total) * 100 if total > 0 else 0
        # Progress output handled by tqdm or other progress bars
        pass
        if current >= total:
            pass  # Progress complete


def format_metrics(metrics_dict):
    """Format metrics dictionary for pretty printing"""
    formatted = []
    for key, value in metrics_dict.items():
        formatted.append(f"{key}: {value:.4f}")
    return ", ".join(formatted)