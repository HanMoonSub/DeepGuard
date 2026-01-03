import torch
import torch.nn as nn
from .layers.featextractor import FeatExtractor
from .layers.transformer import ViTLayer
from typing import List, Union

class MultiScaleEffViT(nn.Module):
    def __init__(
            self,
            model_name: str,
            img_size: List[int],
            l_block_idx: int, 
            h_block_idx: int,
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
            **kwargs
            ):
        super().__init__()
        
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
        return {'pos_embed','cls_token'}
    
    def forward(self, x):
        
        feat = self.feat_extractor(x) # (B,C,H,W)
        
        l_feat = feat[self.l_meta["feature_idx"]]
        h_feat = feat[self.h_meta["feature_idx"]]
        
        l_out = self.l_vit(l_feat) # (B,num_classes)
        h_out = self.h_vit(h_feat) # (B,num_classes)
    
        return l_out + h_out