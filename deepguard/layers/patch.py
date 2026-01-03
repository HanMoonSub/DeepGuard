import torch
import torch.nn as nn
from typing import Type, Optional, List
from .reducesize import ReduceSize
from .se import SE
from .weight_init import trunc_normal_

class LocalContextBlock(nn.Module):
    
    """
        MBConv -> MaxPool 2X2 (Optional)
        주변 픽셀 간의 상관관계를 학습하고 해상도를 조절하는 블록
    """
    def __init__(self, 
                 dim: int,
                 keep_dim: bool = False):
        super().__init__()
        
        self.keep_dim = keep_dim
        
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if not self.keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x = x.contiguous()
        x =  x + self.mb_conv(x) # skip connection
        if not self.keep_dim:
            x = self.pool(x)
            
        return x

class LowBlockPatchEmbed(nn.Module):
    
    """
    Feature map: block.0 output, (B,C,H,W) -> (B,H//8,W//8,D)
    Feature map: block.1 output, (B,C,H,W) -> (B,H//4,W//4,D)
    Feature map: block.2 output, (B,C,H,W) -> (B,H//2,W//2,D)
    
    """
    
    def __init__(
                self,
                in_chs: int,
                dim: int,
                block_idx: int,
                input_resolution: List[int]
                ):
        super().__init__()
        
        """
        Args:
            in_chs: Feature Map Channels
            dim: Embedding Dimension
            block_idx: Backbone Block index
            input_resolution: Feature map Height, Width
        """
        
        H, W = input_resolution
        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        
        if block_idx == 0:
            self.proj = nn.Conv2d(in_chs, dim, kernel_size=7, stride=4, padding=3)
            n_h, n_w = H // 8, W // 8
        elif block_idx == 1:
            self.proj = nn.Conv2d(in_chs, dim, kernel_size=3, stride=2, padding=1)
            n_h, n_w = H // 4, W // 4
        elif block_idx == 2:
            self.proj = nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0)
            n_h, n_w = H // 2, W // 2
        
        self.pos_embed = nn.Parameter(torch.zeros(1, n_h * n_w + 1, dim))    
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)
        
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
        x = self.conv_down(x)
        
        B,H,W,C = x.shape
        x = x.reshape(B,-1,C)
        
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_token, x), dim=1)
        
        x += self.pos_embed
        
        return x # (B,N+1,C)
    
    
class HighBlockPatchEmbed(nn.Module):
    
    """
    Feature map: block.4 output, (B,C,H,W) -> (B,H//4,W//4,D)
    Feature map: block.6 output, (B,C,H,W) -> (B,H,W,D)
    
    """
    
    def __init__(
                self,
                in_chs: int,
                dim: int, 
                block_idx: int, 
                input_resolution: List[int]
                ):
        super().__init__()
        
        """
        Args:
            in_chs: Feature Map Channels
            dim: Embedding Dimension
            block_idx: Backbone Block index
            input_resolution: Feature map Height, Width
        """
        
        H, W = input_resolution
        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        
        if block_idx == 4:
            self.block = nn.Sequential(
                LocalContextBlock(in_chs),
                LocalContextBlock(in_chs),
                nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0)
            )
            n_h, n_w = H // 4, W // 4
        
        elif block_idx == 6:
            self.block = nn.Sequential(
                LocalContextBlock(in_chs, keep_dim=True),
                nn.Conv2d(in_chs, dim, kernel_size=1, stride=1, padding=0)
            )
            n_h, n_w = H, W
            
        self.pos_embed = nn.Parameter(torch.zeros(1, n_h * n_w + 1, dim))    
        
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        
        x = self.block(x) # (B,in_chs,H,W) -> (B,D,H,W) or (B,D,H//2,W//2)
        x = x.permute(0,2,3,1).contiguous() # (B,C,H,W) -> (B,H,W,C)
        
        B,H,W,C = x.shape
        x = x.reshape(B,-1,C) # (B,N,C)
        
        cls_token = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_token, x), dim=1)    
            
        x += self.pos_embed
        
        return x # (B,N+1,C)