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
        
        with torch.no_grad():
            dummy_input = torch.zeros(1,3,*img_size)
            l_result, h_result = self.feat_extractor(dummy_input)
        
        self.l_vit = ViTLayer(
                            in_chs = l_result['in_chs'], 
                            block_idx = l_result['block_idx'], 
                            input_resolution = l_result['input_resolution'], 
                            dim = l_dim, 
                            depth = l_depth, 
                            num_heads = l_heads, 
                            mlp_ratio = l_ratio,
                            drop = l_drop,
                            attn_drop = l_attn_drop,
                            drop_path = l_drop_path,
                            patch_embed = 'low')
        
        self.h_vit = ViTLayer(
                            in_chs = h_result['in_chs'], 
                            block_idx = h_result['block_idx'], 
                            input_resolution = h_result['input_resolution'], 
                            dim = h_dim, 
                            depth = h_depth, 
                            num_heads = h_heads, 
                            mlp_ratio = h_ratio,
                            drop = h_drop,
                            attn_drop = h_attn_drop,
                            drop_path = h_drop_path,
                            patch_embed = 'high')
        
    def forward(self, x):
        
        l_result, h_result = self.feat_extractor(x) # (B,C,H,W)
        
        l_out = self.l_vit(l_result['feature_map']) # (B,num_classes)
        h_out = self.h_vit(h_result['feature_map']) # (B,num_classes)
    
        return l_out + h_out