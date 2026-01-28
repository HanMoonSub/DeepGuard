import torch
import torch.nn as nn
from .attention import MSA
from typing import Optional, Type, List
from .mlp import Mlp
from .drop import DropPath
from .patch import PatchEmbed
from .weight_init import trunc_normal_

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
    
class ViTLayer(nn.Module):
    def __init__(
                self,
                in_chs: int,
                block_idx: int,
                input_resolution: List[int], 
                dim: int,
                depth: int,
                num_heads: int,
                mlp_ratio: float,
                qkv_bias: bool = True,
                qk_scale: Optional[float] = None,
                drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.2,
                act_layer: Type[nn.Module] = nn.GELU,
                attention = MSA,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                layer_scale: float = None,
                **kwargs
    ):
        """
        Args:
            in_chs: feature map channels
            block_idx: backbone block index
            input_resolution: feature map height, width
            dim: feature size dimension.
            depth: number of transformer blocks
            num_heads: number of heads in each stage.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: mlp dropout, proj dropout, pos dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            act_layer: activation function type
            attention: Multi Head Self-Attention type
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        
        super().__init__()
        
        self.patch_embed = PatchEmbed(in_chs, dim, block_idx, input_resolution)
        self.pos_drop = nn.Dropout(p=drop)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = ViTBlock(
                dim = dim,
                num_heads = num_heads,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
                drop = drop,
                attn_drop = attn_drop,
                drop_path = dpr[i],
                act_layer = act_layer,
                attention = attention,
                norm_layer = norm_layer,
                layer_scale = layer_scale,
            )
            self.blocks.append(block)
            
        self.norm = norm_layer(dim)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.0)
    
    
    def forward(self, x):
        
        x = self.patch_embed(x) # (B, in_chs, H, W) -> (B, n_w * n_h, dim)
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x) # (B, n_w * n_h, dim)
            
        x = self.norm(x) # (B, n_w * n_h, dim)
        
        return x