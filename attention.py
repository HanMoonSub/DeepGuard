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
    
## Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, kernel_size=3):
        super(SAM, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        cat = torch.cat([max_out, avg_out], dim=1)

        out = self.conv(cat)
        
        return x * self.sigmoid(out)

## Channel Attention Module
class CAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(self.in_planes, self.in_planes//self.ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes//self.ratio, self.in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out
        return x * self.sigmoid(out)

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


## Efficient Channel Attention Module
class ECA(nn.Module):
    def __init__(self, in_planes, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log2(in_planes) / gamma) + b))
        k = t if t % 2 else t + 1  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.conv = nn.Conv1d(2, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        
        avg_out = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        max_out = self.max_pool(x).squeeze(-1).transpose(-1, -2)  # (B, 1, C)
        
        out = torch.cat([avg_out, max_out], dim=1)

        out = self.conv(out)   # (B, 1, C)
        out = self.sigmoid(out).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)

        return x * out
    
    
## Attention Gate(HxW, CxH, CxW)
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
   
        self.conv = nn.Sequential(
            nn.Conv2d(2,1, kernel_size=3, stride=1, padding="same", bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B,1,H,W)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B,1,H,W)

        cat = torch.cat([max_out, avg_out], dim=1)

        x_out = self.conv(cat)

        return x * self.sigmoid(x_out)

## Triplet Attention Module
class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        
        self.cw = AttentionGate() # Channel, Width
        self.hc = AttentionGate() # Height, Channel
        self.hw = AttentionGate() # Height, Width

    def forward(self, x):
        # x: (Batch, Channel, Height, Width)
        
        # Channel, Width
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out1 = x_out1.permute(0,2,1,3).contiguous()

        # Height, Channel
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out2 = x_out2.permute(0,3,2,1).contiguous()

        # Height, Width
        x_out = self.hw(x)
        x_out = (x_out + x_out1 + x_out2) / 3

        return x_out