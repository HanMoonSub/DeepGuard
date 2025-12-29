import torch
import torch.nn as nn
from typing import Optional

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
