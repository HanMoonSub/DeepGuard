import torch
import torch.nn as nn
from deepguard.layers.featextractor import FeatExtractor
from deepguard.layers.transformer import ViTLayer
from deepguard.layers.weight_init import trunc_normal_
from typing import List, Union
from timm.models import register_model, build_model_with_cfg

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1, # Real or Fake 
        'input_size': (3, 224, 224), 
        'pool_size': None,
        'crop_pct': 0.875, 
        'interpolation': 'bicubic', 
        'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406), 
        'std': (0.229, 0.224, 0.225),
        **kwargs
    }
    
default_cfgs = {
    'ms_eff_vit_b0': _cfg(
            input_size = (3,224,224)
        ),
    'ms_eff_vit_b5': _cfg(
            input_size = (3,384,384)
        ),
}

weight_registry = {
    'ms_eff_vit_b0': {
        'asian':    'https://github.com/.../releases/download/v0.2.0/b0_asian.pth',
        'western1': 'https://github.com/.../releases/download/v0.1.0/b0_western_v1.pth',
        'western2': 'https://github.com/.../releases/download/v0.3.0/b0_western_v2.pth',
    },
    'ms_eff_vit_b5': {
        'asian':    'https://github.com/.../releases/download/v0.2.0/b5_asian.pth',
        'western1': 'https://github.com/.../releases/download/v0.1.0/b5_western_v1.pth',
        'western2': 'https://github.com/.../releases/download/v0.3.0/b5_western_v2.pth',
    }
}
class MultiScaleEffViT(nn.Module):
    def __init__(
            self,
            model_name: str,
            img_size: List[int],
            l_block_idx: int = 1, 
            h_block_idx: int = 6,
            l_dim: int = 256,
            h_dim: int = 512, 
            l_depth: int = 2,
            h_depth: int = 4,
            l_heads: int = 4,
            h_heads: int = 8, 
            l_ratio: Union[float, int] = 4.,
            h_ratio: Union[float, int] = 4.,
            l_drop: float = 0.,
            h_drop: float = 0.,
            l_attn_drop: float = 0.,
            h_attn_drop: float = 0.,
            l_drop_path: float = 0.,
            h_drop_path: float = 0.,
            num_classes: int = 1,
            pool: str = 'avg', # or 'str'
            **kwargs
            ):
        super().__init__()
        
        self.pool = pool
        self.feat_extractor = FeatExtractor(
                            model_name = model_name, 
                            img_size = img_size, 
                            l_block_idx = l_block_idx, 
                            h_block_idx = h_block_idx
                            )
        
        self.l_meta = self._build_metadata(img_size, l_block_idx)
        self.h_meta = self._build_metadata(img_size, h_block_idx)
        
        self.l_vit = ViTLayer(
                            in_chs = self.l_meta['in_chs'], 
                            block_idx = self.l_meta['block_idx'], 
                            input_resolution = self.l_meta['input_resolution'], 
                            dim = l_dim, 
                            depth = l_depth, 
                            num_heads = l_heads, 
                            mlp_ratio = l_ratio,
                            drop = l_drop,
                            attn_drop = l_attn_drop,
                            drop_path = l_drop_path,
                            patch_embed = 'low')
        
        self.h_vit = ViTLayer(
                            in_chs = self.h_meta['in_chs'], 
                            block_idx = self.h_meta['block_idx'], 
                            input_resolution = self.h_meta['input_resolution'], 
                            dim = h_dim, 
                            depth = h_depth, 
                            num_heads = h_heads, 
                            mlp_ratio = h_ratio,
                            drop = h_drop,
                            attn_drop = h_attn_drop,
                            drop_path = h_drop_path,
                            patch_embed = 'high')
        
        fusion_dim = h_dim + l_dim
        self.head = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, num_classes)
            ) if num_classes > 0 else nn.Identity()
        
        self._init_head_weights()
        
    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)
        
    def _build_metadata(self, img_size, block_idx):
        
        if block_idx == 6:
            feature_idx = 4
        elif block_idx == 4:
            feature_idx = 3
        else:
            feature_idx = block_idx
        
        info = self.feat_extractor.backbone.feature_info[feature_idx]
        
        meta = {
            "block_idx": block_idx,
            "feature_idx": feature_idx,
            "in_chs": info['num_chs'],
            "input_resolution": [img_size[0] // info['reduction'], img_size[1] // info['reduction']]
        }
        
        return meta
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'pos_embed','cls_token','bn','norm'}
    
    def forward(self, x):
        
        feat = self.feat_extractor(x) # (B,C,H,W)
        
        l_feat = feat[self.l_meta["feature_idx"]]
        h_feat = feat[self.h_meta["feature_idx"]]
        
        l_out = self.l_vit(l_feat) # (B,N_l+1,l_dim)
        h_out = self.h_vit(h_feat) # (B,N_h+1,h_dim)
        
        if self.pool == 'cls':
            l_pool = l_out[:,0]
            h_pool = h_out[:,0]
        elif self.pool == "avg":
            l_pool = l_out[:,1:].mean(dim=1)
            h_pool = h_out[:,1:].mean(dim=1)
        
        fusion_out = torch.concat([l_pool, h_pool], dim=-1)
            
        return self.head(fusion_out)
    
def _get_config_for_type(variant: str, type_key: str) -> dict:
   
    config = default_cfgs[variant].copy()
    
    urls = weight_registry.get(variant, {})
    if type_key not in urls:
        available_types = list(urls.keys())
        raise ValueError(f"Type '{type_key}' not found for {variant}. Available: {available_types}")
    
    config['url'] = urls[type_key]
    return config

def _create_ms_eff_vit(variant,  
                       pretrained_cfg,
                       pretrained=False,
                       **kwargs):
    
    return build_model_with_cfg(
        MultiScaleEffViT,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        **kwargs
    )
    

@register_model
def ms_eff_vit_b0(pretrained=False, domain="asian", **kwargs) -> MultiScaleEffViT:

    variant = "ms_eff_vit_b0"
    
    kwargs.pop('pretrained_cfg', None)
    
    model_kwargs = dict(
        model_name = "tf_efficientnet_b0.ns_jft_in1k",
        img_size = [224,224],
        l_dim = 120,
        h_dim = 160, 
        l_depth = 2,
        h_depth = 6,
        l_heads = 4,
        h_heads = 4,
        l_ratio = 4,
        h_ratio = 6,
        h_drop = 0.1,
        h_drop_path = 0.1,
        **kwargs
    )
    
    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, domain)
    else:
        pretrained_cfg = default_cfgs[variant]
        
    return _create_ms_eff_vit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained, 
                              **model_kwargs)

@register_model
def ms_eff_vit_b5(pretrained=False, domain="asian", **kwargs) -> MultiScaleEffViT:
    
    variant = "ms_eff_vit_b5"
    
    kwargs.pop('pretrained_cfg', None)
    
    model_kwargs = dict(
        model_name = "tf_efficientnet_b5.ns_jft_in1k",
        img_size = [384,384],
        l_dim = 320,
        h_dim = 512, 
        l_depth = 3,
        h_depth = 6,
        l_heads = 8,
        h_heads = 8, 
        l_ratio = 4,
        h_ratio = 4,
        l_drop = 0.0,
        h_drop = 0.1,
        l_attn_drop = 0.05,
        h_attn_drop = 0.05,
        l_drop_path = 0.05,
        h_drop_path = 0.1,
        **kwargs
    )
    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, domain)
    else:
        pretrained_cfg = default_cfgs[variant]
        
    return _create_ms_eff_vit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained, 
                              **model_kwargs)
