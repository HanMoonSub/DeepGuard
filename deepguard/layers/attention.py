import torch
import torch.nn as nn
from typing import Optional
from .weight_init import trunc_normal_

class MSA(nn.Module):
    def __init__(
                self,
                dim: int,
                num_heads: int,
                qkv_bias: bool = True,
                qk_scale: Optional[float] = None,
                attn_drop: float = 0.,
                proj_drop: float = 0.
                ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        B, N, C = x.shape # (B, num_patches, Embedding dim)
        
        # (3, B, num_heads, num_patches, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LocalWindowAttention(nn.Module):
    """
        inp: (B*num_windows, window_size**2, dim)
        out: (B*num_windows, window_size**2, dim)
    """
    
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.
                 ):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        
        # ========= Learnable Bias Table ============
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1)*(2*window_size-1),num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:,:,None] - coords_flatten[:,None,:]
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += window_size - 1
        relative_coords[:,:,1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1 # h_idx = h_dix * width
        relative_position_index = relative_coords.sum(-1) 
        
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, q_global):
        B, N, C = x.shape # (B*num_windows, window_size**2, dim)
        # (B*num_windows, N, 3, num_heads, head_dim) -> (3, B*num_windows, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        # q: query, k: key, v: value
        q, k, v = qkv[0], qkv[1], qkv[2] # (B*num_windows, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1)) # (B*num_windows, num_heads, N, N)
        
        # ((2W-1)*(2W-1), num_heads) -> (w*w, w*w, num_heads)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size**2, self.window_size**2, -1
        )
        
        # (w*w, w*w, num_heads) -> (num_heads, N, N)
        relative_position_bias = relative_position_bias.permute(2,0,1).contiguous()
        
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # (B*num_heads, num_heads, N, head_dim) -> (B*num_heads, N, C)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x