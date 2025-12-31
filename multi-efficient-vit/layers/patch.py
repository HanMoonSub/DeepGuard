import torch.nn as nn
from typing import Type, Optional
from .reducesize import ReduceSize
        
class PatchEmbed(nn.Module):
    def __init__(
                self,
                in_chs: int,
                dim: int,
                ):
        super().__init__()
        self.proj = nn.Conv2d(in_chs, dim, kernel_size=3, stride=2, padding=1)
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
        x = self.conv_down(x)
        
        return x