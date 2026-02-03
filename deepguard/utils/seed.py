import os
import random
import numpy as np
import torch



def seed_everything(SEED: int):
    
    """Reproducibility를 위해 모든 랜덤 시드 고정"""
    
    # Python & NumPy & Torch
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # cudnn 설정 (재현성 향상)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python 해시 시드 고정
    os.environ['PYTHONHASHSEED'] = str(SEED)