import torch
import math
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau

def fetch_scheduler(optimizer, cfg):
    """
    Builds a learning rate scheduler based on the training configuration.
    
    Supports:
    1. CosineAnnealingLR (Custom Ratio-based)
    2. StepLR
    3. ReduceLROnPlateau
    """
    
    if cfg.train.scheduler == 'CosineAnnealingLR':
        # Every parameter group's LR will drop to (Initial LR * 1e-3)
        # This ensures proportional decay regardless of the starting LR of each component.
        min_lr_ratio = 1e-3 
        total_epochs = cfg.train.epochs
        
        cosine_fn = lambda epoch: min_lr_ratio + 0.5 * (1 - min_lr_ratio) * \
                                  (1 + math.cos(math.pi * epoch / total_epochs))
        
        # LambdaLR applies the cosine_fn to the initial_lr of EACH parameter group.
        # This effectively sets a different eta_min for backbone, l_vit, and h_vit.
        scheduler = LambdaLR(optimizer, lr_lambda=cosine_fn)
        
        return scheduler

    elif cfg.train.scheduler == 'StepLR':
        # Decays the learning rate of each parameter group by gamma every step_size epochs.
        # Default: drops to 10% (gamma=0.1) every 10% of total epochs.
        scheduler = StepLR(
            optimizer, 
            step_size=max(1, cfg.train.epochs // 10), 
            gamma=0.1, 
            last_epoch=-1
        )
        return scheduler

    elif cfg.train.scheduler == 'ReduceLROnPlateau':
        # Reduces learning rate when a metric (e.g., validation loss) has stopped improving.
        # mode='min': monitor if the value is no longer decreasing.
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=1, 
            factor=0.1, 
            threshold=0.1, 
            verbose=True # Helpful to see LR changes in console
        )
        return scheduler

    else:
        # Error handling for unsupported scheduler types
        raise ValueError(
            f"❌ Unsupported scheduler type: {cfg.train.scheduler}\n"
            f"Supported types are: ['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau']"
        )