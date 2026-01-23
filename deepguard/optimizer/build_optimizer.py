import torch

def add_to_optim_groups(skip_keywords, 
                        params_list, 
                        base_lr, 
                        base_wd):
    
    """
    Filters parameters into decay and no-decay groups.
    
    Args:
        skip_keywords (set): Keywords for parameters that should not have weight decay.
        params_list (list): List of (name, param) tuples.
        base_lr (float): Learning rate for this specific component.
        base_wd (float): Weight decay rate for this specific component.
    """
    
    optim_groups = []
    decay = []
    no_decay = []
    
    for name, param in params_list:
        if not param.requires_grad:
            continue
        
        if any(key in name for key in skip_keywords) or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    
    if decay:
        optim_groups.append({'params': decay, 'lr': base_lr, 'weight_decay': base_wd})
    if no_decay:
        optim_groups.append({'params': no_decay, 'lr': base_lr, 'weight_decay': 0.0})

    return optim_groups

def build_ms_eff_vit_optimizer(model, cfg):
    """
    Constructs an optimizer with differential learning rates and weight decay exclusion.
    
    This builder segments the Multi-Scale EffViT parameters into three main components:
    1. Feature Extractor (Backbone)
    2. Low-resolution ViT blocks (L-ViT)
    3. High-resolution ViT blocks (H-ViT)
    """
    
    skip_keywords = model.no_weight_decay_keywords()
    
    feat_params = []
    l_vit_params = []
    h_vit_params = []
    
    for name, param in model.named_parameters():
        if "feat_extractor" in name:
            feat_params.append((name, param))
        elif "l_vit" in name:
            l_vit_params.append((name, param))
        elif "h_vit" in name:
            h_vit_params.append((name, param))
            
    all_optim_groups = []
    
    all_optim_groups.extend(
        add_to_optim_groups(skip_keywords, feat_params, cfg.train.backbone_lr, cfg.train.backbone_wd)
    )
    all_optim_groups.extend(
        add_to_optim_groups(skip_keywords, l_vit_params, cfg.train.l_block_lr, cfg.train.l_block_wd)
    )
    all_optim_groups.extend(
        add_to_optim_groups(skip_keywords, h_vit_params, cfg.train.h_block_lr, cfg.train.h_block_wd)
    )
    opt_class = getattr(torch.optim, cfg.train.optimizer)
    return opt_class(all_optim_groups)