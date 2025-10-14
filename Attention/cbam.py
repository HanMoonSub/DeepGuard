import torch
import torch.nn as nn
import math
from .cam import CAM
from .sam import SAM

## Convolutional Block Attention Module
class CBAMBlock(nn.Module):
    def __init__(self, in_planes):
        super(CBAMBlock, self).__init__()
        self.sam = SAM()
        self.cam = CAM(in_planes)

    def forward(self, x):
        cam_out = self.cam(x)
        sam_out = self.sam(cam_out)

        return sam_out