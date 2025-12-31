import torch.nn as nn
from typing import Type
from .se import SE

class ReduceSize(nn.Module):
    def __init__(
                self,
                dim: int,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                keep_dim: bool = False
                ):
        super().__init__()
        
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim bias=False)
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
            
        self.reduction = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2,
                                   padding=1, bias = False)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim_out)
        
    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = x.permute(0,3,1,2) # (B,H,W,C) -> (B,C,H,W)
        x = x + self.mb_conv(x)
        x = x.permute(0,2,3,1)
        x = self.norm2(x)
        return x