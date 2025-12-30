import torch
import torch.nn as nn
from .attention import MSA
from typing import Optional, Type
from .mlp import Mlp
from .drop import DropPath

class ViTBlock(nn.Module):
    def __init__(
                self,
                dim: int,
                num_heads: int,
                mlp_ratio: float = 4.,
                qkv_bias: bool = True,
                qk_scale: Optional[float] = None,
                drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                act_layer: Type[nn.Module] = nn.GELU,
                attention = MSA,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                layer_scale: Optional[float] = None,
                ):
        super().__init__()
        # Layer Normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.attn = attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop)
    
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_dims=dim, hidden_dims=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
    
        self.layer_scale = False
        if layer_scale is not None:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
            
        
    def forward(self, x):
        
        shortcut = x # (B, N, C)
        x = self.norm1(x)
        attn = self.attn(x)
        
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        
        return x