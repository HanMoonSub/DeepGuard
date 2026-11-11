import os
import random
import numpy as np
import torch



def seed_everything(SEED: int):
    
    """Fix all random seeds for reproducibility"""
    
    # Python & NumPy & Torch
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # cudnn settings (improves reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fix the Python hash seed
    os.environ['PYTHONHASHSEED'] = str(SEED)