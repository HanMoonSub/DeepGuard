import timm
import torch
import torch.nn as nn
from typing import List, Optional

class FeatExtractor(nn.Module):
    """
    A multi-scale feature extraction module using an EfficientNet backbone.
    
    It extracts 'Subtle Artifacts' from low-level blocks and 
    'Global Features' from high-level blocks for DeepFake detection tasks.
    """
    
    def __init__(
                self,
                model_name: str,
                img_size: List[int],
                l_block_idx: int, # Index for low-level feature extraction (e.g., 0, 1, 2)
                h_block_idx: int, # Index for high-level feature extraction (e.g., 4, 6)
                pretrained: bool = True,
    ):
        super().__init__()
        
        """
        Args:
            model_name (str): Name of the model to create via timm (e.g., 'efficientnet_b5').
            img_size (List[int]): Input image resolution as [H, W].
            l_block_idx (int): Target low-level block index 
            h_block_idx (int): Target high-level block index 
            pretrained (bool): Whether to use default ImageNet pretrained weights.
        """
        
        # When features_only=True, timm typically returns 5 feature maps (reduction 2, 4, 8, 16, 32).
        # Reference for EfficientNet-B5:
        # Index 0 (Stage 1): blocks.0 | Reduc 2  | chs 16
        # Index 1 (Stage 2): blocks.1 | Reduc 4  | chs 24
        # Index 2 (Stage 3): blocks.2 | Reduc 8  | chs 40
        # Index 3 (Stage 5): blocks.4 | Reduc 16 | chs 112
        # Index 4 (Stage 7): blocks.6 | Reduc 32 | chs 320
        
        self.backbone = timm.create_model(
            model_name,
            pretrained = pretrained,
            features_only = True,
        )
        
            
        if l_block_idx not in (0, 1, 2): self.l_block_idx = l_block_idx
        else: raise ValueError("l_block_idx must be 0, 1, or 2.")
        
        if h_block_idx in (4, 6): self.h_block_idx = h_block_idx
        else: raise ValueError("h_block_idx must be 4 or 6.")
                            
        self.l_meta = self._build_metadata(img_size, self.l_block_idx)
        self.h_meta = self._build_metadata(img_size, self.h_block_idx)
        
    def _build_metadata(self, img_size, block_idx):
        
        if block_idx == 6:
            feature_idx = 4
        elif block_idx == 4:
            feature_idx = 3
        else:
            feature_idx = block_idx
        
        info = self.backbone.feature_info[feature_idx]
        
        meta = {
            "block_idx": block_idx,
            "feature_idx": feature_idx,
            "in_chs": info['num_chs'],
            "input_resolution": [img_size[0] // info['reduction'], img_size[1] // info['reduction']]
        }
        
        return meta
        
    def forward(self, x):
    
        out = self.backbone(x)

        l_out = out[self.l_meta['feature_idx']]
        h_out = out[self.h_meta['feature_idx']]
        
            
        l_result = {**self.l_meta, "feature_map": l_out}
        h_result = {**self.h_meta, "feature_map": h_out} 
            
        return l_result, h_result
