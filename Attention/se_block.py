import torch
import torch.nn as nn
import math

## Squeeze and Excitation Module
class SEBlock(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SEBlock, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(self.in_planes, self.in_planes//self.ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes//self.ratio, self.in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.avg_pool(x)
        out = self.fc(out)

        return x * self.sigmoid(out)