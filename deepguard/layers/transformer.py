import torch
import torch.nn as nn
from typing import Optional, Type, List
from .attention import MSA, GlobalWindowAttention, LocalWindowAttention
from .mlp import Mlp
from .drop import DropPath
from .patch import PatchEmbed, GCViTPatchEmbed
from .global_query import GlobalQueryGen
from .reducesize import ReduceSize
from .window import window_partition, window_reverse
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
    
class GCViTBlock(nn.Module):
    """
        inp: (B,H,W,C)
        out: (B,H,W,C)
        
        if attention: 'local'
            LocalWindowAttention
        if attention: 'global'
            GlobalWindowAttention
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 attention: Type[nn.Module] = LocalWindowAttention,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 layer_scale: int | float | None = None                 
                 ):
        """Args
            dim: feature size dimension
            num_heads: number of attention head
            window_size: window_size
            mlp_ratio: MLP ratio
            qkv_bias: bool argument for query, key, value learnable bias
            qk_scale: bool argument to scaling query, key
            drop: proj, mlp dropout ratio
            attn_drop: attention dropout ratio
            drop_path: drop path rate
            act_layer: activation function
            attention: attention block type
            norm_layer: normalization layer
            layer_scale: layer scaling coefficient
        """
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.attn = attention(
            dim,
            num_heads = num_heads,
            window_size = window_size,
            qkv_bias = qkv_bias,
            qk_scale = qk_scale,
            attn_drop = attn_drop,
            proj_drop = drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = Mlp(in_dims=dim, hidden_dims=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        
        self.layer_scale = False
        if layer_scale is not None:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
    
    def forward(self, x, q_global):
        B,H,W,C = x.shape
        shortcut = x
        x = self.norm1(x)
        # (B, H, W, C) -> (B*num_windows, window_size, window_size, C)
        x_windows = window_partition(x, self.window_size)
        # (B*num_windows, window_size, window_size, C) -> (B*num_windows, window_size**2, C)
        x_windows = x_windows.view(-1, self.window_size**2, C)
        attn_windows = self.attn(x_windows, q_global)
        # (B*num_windows, window_size ** 2, C) -> (B, H, W, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        
        return x
    
class GCViTLayer(nn.Module):
    """
        inp: (B,H,W,C)
        out: 
            if downsample:
                (B,H//2,W//2,2C)
            else:
                (B,H,W,C)
    """
    def __init__(self,
                 dim: int,
                 depth: int,
                 input_resolution: int | float,
                 num_heads: int,
                 window_size: int,
                 downsample: bool = True,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: int | float | None = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: List[float] | float = 0.,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 layer_scale: int | float | None = None,
                 ):
        """
        Args:
            dim: feature size dimension
            depth: number of block in each stage
            input_resolution: input image resolution
            num_heads: number of heads in each stage
            window_size: window size in each stage
            downsample: bool argument for down-sampling
            mlp_ratio: MLP ratio
            qkv_bias: bool argument for query, key, value learnable bias
            qk_scale: bool argument for scaling query, key
            drop: proj, mlp dropout rate
            attn_drop: attention dropout rate
            drop_path: drop path rate
            norm_layer: normalization layer
            layer_scale: layer scaling coefficient
        """
        
        
        super().__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(
                dim = dim,
                num_heads = num_heads,
                window_size = window_size,
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
                drop = drop,
                attn_drop = attn_drop,
                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                attention = LocalWindowAttention if (i % 2 == 0) else GlobalWindowAttention,
                norm_layer = norm_layer,
                layer_scale = layer_scale,
            )
            for i in range(depth)
        ])
        
        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        self.q_global_gen = GlobalQueryGen(dim, input_resolution, window_size, num_heads)
    def forward(self, x):
        # (B,H,W,C) -> (B,C,H,W)
        q_global = self.q_global_gen(x.permute(0,3,1,2))
        
        for block in self.blocks:
            x = block(x, q_global)
        
        if self.downsample is None:
            return x
        # (B,H,W,C) -> (B,H//2,W//2,2C)
        return self.downsample(x)
    
class GCViT(nn.Module):
    """
        inp: (B,in_chs,H,W)
        out: (B,C)
    """
    def __init__(self,
                in_chs: int, 
                input_resolution: int | float,
                dim: int, 
                depths: List[int],
                window_size: List[int],
                mlp_ratio: List[float],
                num_heads: List[int], 
                qkv_bias: bool = True,
                qk_scale: int | float | None = None,
                drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                layer_scale: int | float | None = None,
                **kwargs):
        """
        Args:
            in_chs: backbone feature map channels
            input_resolution: backbone feature map size
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio in each stage
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        
        super().__init__()
        num_feats = int(dim * 2 ** (len(depths)-1))
        self.patch_embed = GCViTPatchEmbed(in_chs = in_chs, dim = dim)
        self.pos_drop = nn.Dropout(drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        
        self.stages = nn.ModuleList([
            GCViTLayer(
                dim = int(dim * 2 ** i),
                depth = depths[i],
                input_resolution = int(input_resolution * 2 ** (-i)),
                num_heads = num_heads[i],
                window_size = window_size[i],
                downsample = (i < len(depths) - 1),
                mlp_ratio = mlp_ratio[i],
                qkv_bias = qkv_bias,
                qk_scale = qk_scale,
                drop = drop, 
                attn_drop = attn_drop,
                drop_path = dpr[sum(depths[:i]):sum(depths[:i+1])],
                norm_layer = norm_layer,
                layer_scale = layer_scale,
            )
            for i in range(len(depths))
        ])
        
        self.norm = norm_layer(num_feats)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    
    def forward(self, x):
        x = self.patch_embed(x) #(B,in_chs,H,W) -> (B,H,W,D)
        x = self.pos_drop(x)
        
        for stage in self.stages:
            x = stage(x) # (B,H,W,D) -> (B,H//2,W//2,2D)
            
        x = self.norm(x)
        x = x.permute(0,3,1,2) # (B,h,w,C) -> (B,C,h,w)
        x = self.avgpool(x) # (B,C,1,1)
        x = x.reshape(x.shape[0],-1) #(B,C,1,1) -> (B,C)
        
        return x