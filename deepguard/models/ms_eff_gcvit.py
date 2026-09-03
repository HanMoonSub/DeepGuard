import torch
import torch.nn as nn
from typing import Type
from deepguard.layers.featextractor import FeatExtractor
from deepguard.layers.transformer import GCViT
from deepguard.layers.weight_init import trunc_normal_
from typing import List, Union
from timm.models import register_model, build_model_with_cfg

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1, # Real or Fake 
        'input_size': (3, 224, 224), 
        'pool_size': None,
        'crop_pct': 0.875, 
        'interpolation': 'bicubic', 
        'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406), 
        'std': (0.229, 0.224, 0.225),
        **kwargs
    }
    
default_cfgs = {
    'eff_b0': _cfg(
                input_size = (3,224,224)
        ),
    'eff_b5': _cfg(
                input_size = (3,384,384)
        ),
    'low_eff_gcvit_b0': _cfg(
                    input_size = (3,224,224)
        ),
    'low_eff_gcvit_b5': _cfg(
                    input_size = (3,384,384)
        ),
    'high_eff_gcvit_b0': _cfg(
                    input_size = (3,224,224)
        ),
    'high_eff_gcvit_b5': _cfg(
                    input_size = (3,384,384)
        ),   
    'ms_eff_gcvit_b0': _cfg(
            input_size = (3,224,224)
        ),
    'ms_eff_gcvit_b5': _cfg(
            input_size = (3,384,384)
        ),
}

weight_registry = {
    'eff_b0': {
            'celeb_df_v2':  'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/eff_b0_celeb_df_v2.bin',
            'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/eff_b0_ff++.bin',
            'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/eff_b0_kodf.bin'
    },
    'eff_b5': {
            'celeb_df_v2':   'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/eff_b5_celeb_df_v2.bin',
            'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/eff_b5_ff++.bin',
            'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/eff_b5_kodf.bin'
    },
    'low_eff_gcvit_b0': {
                'celeb_df_v2':  'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/low_eff_gcvit_b0_celeb_df_v2.bin',
                'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/low_eff_gcvit_b0_ff++.bin',
                'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/low_eff_gcvit_b0_kodf.bin'
    },
    'low_eff_gcvit_b5': {
                'celeb_df_v2':   'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/low_eff_gcvit_b5_celeb_df_v2.bin',
                'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/low_eff_gcvit_b5_ff++.bin',
                'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/low_eff_gcvit_b5_kodf.bin'
    },
    'high_eff_gcvit_b0': {
                    'celeb_df_v2':  'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/high_eff_gcvit_b0_celeb_df_v2.bin',
                    'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/high_eff_gcvit_b0_ff++.bin',
                    'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/high_eff_gcvit_b0_kodf.bin'
    },
    'high_eff_gcvit_b5': {
                    'celeb_df_v2':   'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/high_eff_gcvit_b5_celeb_df_v2.bin',
                    'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/high_eff_gcvit_b5_ff++.bin',
                    'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v1.1.0/high_eff_gcvit_b5_kodf.bin'
    },
    'ms_eff_gcvit_b0': {
        'celeb_df_v2':  'https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b0_celeb_df_v2.bin',
        'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b0_ff++.bin',
        'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v0.2.0/ms_eff_gcvit_b0_kodf.bin'
    },
    'ms_eff_gcvit_b5': {
        'celeb_df_v2':   'https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b5_celeb_df_v2.bin',
        'ff++': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b5_ff++.bin',
        'kodf': 'https://github.com/HanMoonSub/DeepGuard/releases/download/v0.2.0/ms_eff_gcvit_b5_kodf.bin'
    }
}

class MultiScaleEffGCViT(nn.Module):
    def __init__(
            self,
            model_name: str,
            img_size: List[int],
            l_block_idx: int = 1,
            h_block_idx: int = 6,
            use_low: bool = True,
            use_high: bool = True,
            l_dim: int = 24,
            h_dim: int = 256, 
            l_depths: List[int] = [2,2,4,2],
            h_depths: List[int] = [4],
            l_windows: List[int] = [7,7,14,7],
            h_windows: List[int] = [7],
            l_heads: List[int] = [1,2,4,8],
            h_heads: List[int] = [4], 
            l_ratio: List[float] = [4,4,4,4],
            h_ratio: List[float] = [4],
            l_drop: float = 0.,
            h_drop: float = 0.05,
            l_attn_drop: float = 0.05,
            h_attn_drop: float = 0,
            l_drop_path: float = 0.1,
            h_drop_path: float = 0.05,
            num_classes: int = 1,
            **kwargs
            ):
        """
        Args:
            model_name: Name of the backbone architecture (e.g., 'efficientnet_b0').
            img_size: Input image resolution as [Height, Width].
            l_block_idx: Index of the backbone block to extract low-level (high-resolution) features.
            h_block_idx: Index of the backbone block to extract high-level (low-resolution) features.
            use_low: Whether to build and run the low-level GCViT branch.
            use_high: Whether to build and run the high-level GCViT branch.
                When both are False, the high-level backbone feature map is global-average-pooled
                and fed to the head directly (EfficientNet-only baseline).
            l_dim: Initial feature dimension for the low-level GCViT branch.
            h_dim: Initial feature dimension for the high-level GCViT branch.
            l_depths: Number of blocks in each stage of the low-level GCViT branch.
            h_depths: Number of blocks in each stage of the high-level GCViT branch.
            l_windows: Window sizes for each stage of the low-level GCViT branch.
            h_windows: Window sizes for each stage of the high-level GCViT branch.
            l_heads: Number of attention heads in each stage of the low-level GCViT branch.
            h_heads: Number of attention heads in each stage of the high-level GCViT branch.
            l_ratio: MLP expansion ratios for each stage of the low-level GCViT branch.
            h_ratio: MLP expansion ratios for each stage of the high-level GCViT branch.
            l_drop: Dropout rate for the low-level branch.
            h_drop: Dropout rate for the high-level branch.
            l_attn_drop: Attention dropout rate for the low-level branch.
            h_attn_drop: Attention dropout rate for the high-level branch.
            l_drop_path: Stochastic depth rate for the low-level branch.
            h_drop_path: Stochastic depth rate for the high-level branch.
            num_classes: Number of classification classes (set to 1 for binary classification).
            **kwargs: Additional keyword arguments for extended model configuration.
        """
        
        super().__init__()
        self.use_low = use_low
        self.use_high = use_high

        self.feat_extractor = FeatExtractor(
                            model_name = model_name,
                            img_size = img_size,
                            l_block_idx = l_block_idx,
                            h_block_idx = h_block_idx
                            )

        self.l_meta = self._build_metadata(img_size, l_block_idx)
        self.h_meta = self._build_metadata(img_size, h_block_idx)

        fusion_dim = 0

        self.l_gcvit = None
        if use_low:
            self.l_gcvit = GCViT(
                                in_chs = self.l_meta['in_chs'],
                                input_resolution = self.l_meta['input_resolution'],
                                dim = l_dim,
                                depths = l_depths,
                                window_size = l_windows,
                                mlp_ratio = l_ratio,
                                num_heads = l_heads,
                                drop = l_drop,
                                attn_drop = l_attn_drop,
                                drop_path = l_drop_path,
                            )
            fusion_dim += l_dim * (2 ** (len(l_depths) - 1))

        self.h_gcvit = None
        if use_high:
            self.h_gcvit = GCViT(
                                in_chs = self.h_meta['in_chs'],
                                input_resolution = self.h_meta['input_resolution'],
                                dim = h_dim,
                                depths = h_depths,
                                window_size = h_windows,
                                mlp_ratio = h_ratio,
                                num_heads = h_heads,
                                drop = h_drop,
                                attn_drop = h_attn_drop,
                                drop_path = h_drop_path,
                            )
            fusion_dim += h_dim * (2 ** (len(h_depths) - 1))

        self.pool = None
        if not use_low and not use_high:
            self.pool = nn.AdaptiveAvgPool2d(1)
            fusion_dim = self.h_meta['in_chs']

        self.head = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim // 2, num_classes)
            ) if num_classes > 0 else nn.Identity()
        
        self._init_head_weights()
        
    def _init_head_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)
        
    def _build_metadata(self, img_size, block_idx):
        
        idx_map = {6: 4, 4: 3}
        feature_idx = idx_map.get(block_idx, block_idx)
        
        info = self.feat_extractor.backbone.feature_info[feature_idx]
        
        meta = {
            "block_idx": block_idx,
            "feature_idx": feature_idx,
            "in_chs": info['num_chs'],
            "input_resolution": img_size[0] // info['reduction']
        }
        return meta
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table','bn','norm'}
    
    def forward(self, x):

        feat = self.feat_extractor(x) # (B,C,H,W)

        outs = []
        if self.use_low:
            l_feat = feat[self.l_meta["feature_idx"]]
            outs.append(self.l_gcvit(l_feat)) # (B,D_l)
        if self.use_high:
            h_feat = feat[self.h_meta["feature_idx"]]
            outs.append(self.h_gcvit(h_feat)) # (B,D_h)

        if outs:
            fusion_out = torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]
        else:
            h_feat = feat[self.h_meta["feature_idx"]]
            fusion_out = self.pool(h_feat).flatten(1) # (B,C)

        return self.head(fusion_out) # (B,fusion_dim) -> (B,1)
    
def _get_config_for_type(variant: str, type_key: str) -> dict:
   
    config = default_cfgs[variant].copy()
    
    urls = weight_registry.get(variant, {})
    if type_key not in urls:
        available_types = list(urls.keys())
        raise ValueError(f"Type '{type_key}' not found for {variant}. Available: {available_types}")
    
    config['url'] = urls[type_key]
    return config

def _create_ms_eff_gcvit(variant,  
                       pretrained_cfg,
                       pretrained=False,
                       **kwargs):
    
    return build_model_with_cfg(
        MultiScaleEffGCViT,
        variant,
        pretrained,
        pretrained_cfg=pretrained_cfg,
        **kwargs
    )
    
@register_model
def ms_eff_gcvit_b0(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:

    variant = "ms_eff_gcvit_b0"
    
    kwargs.pop('pretrained_cfg', None)
    
    model_kwargs = dict(
        model_name = "tf_efficientnet_b0.ns_jft_in1k",
        img_size = [224,224],
        l_dim = 24,
        h_dim = 256, 
        l_depths = [2,2,4,2],
        h_depths = [4],
        l_windows = [7,7,14,7],
        h_windows = [7],
        l_heads = [1,2,4,8],
        h_heads = [4],
        l_ratio = [4,4,4,4],
        h_ratio = [4],
        h_drop = 0.05,
        l_attn_drop = 0.05,
        l_drop_path = 0.1,
        h_drop_path = 0.05,
        **kwargs
    )
    
    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]
        
    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)

@register_model
def ms_eff_gcvit_b5(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:

    variant = "ms_eff_gcvit_b5"

    kwargs.pop('pretrained_cfg', None)

    model_kwargs = dict(
        model_name = "tf_efficientnet_b5.ns_jft_in1k",
        img_size = [384,384],
        l_dim = 48,
        h_dim = 512,
        l_depths = [2,2,6,2],
        h_depths = [6],
        l_windows = [12,12,24,12],
        h_windows = [12],
        l_heads = [2,4,8,16],
        h_heads = [16],
        l_ratio = [3,3,3,3],
        h_ratio = [3],
        h_drop = 0.1,
        l_attn_drop = 0.1,
        l_drop_path = 0.15,
        h_drop_path = 0.1,
        **kwargs
    )
    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]

    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)

@register_model
def eff_b0(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:

    variant = "eff_b0"

    kwargs.pop('pretrained_cfg', None)

    model_kwargs = dict(
        model_name = "tf_efficientnet_b0.ns_jft_in1k",
        img_size = [224,224],
        use_low = False,
        use_high = False,
        **kwargs
    )

    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]

    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)

@register_model
def eff_b5(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:

    variant = "eff_b5"

    kwargs.pop('pretrained_cfg', None)

    model_kwargs = dict(
        model_name = "tf_efficientnet_b5.ns_jft_in1k",
        img_size = [384,384],
        use_low = False,
        use_high = False,
        **kwargs
    )

    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]

    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)

@register_model
def low_eff_gcvit_b0(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:
    
    variant = "low_eff_gcvit_b0"

    kwargs.pop('pretrained_cfg', None)

    model_kwargs = dict(
        model_name = "tf_efficientnet_b0.ns_jft_in1k",
        img_size = [224,224],
        use_low = True,
        use_high = False,
        l_dim = 24,
        l_depths = [2,2,4,2],
        l_windows = [7,7,14,7],
        l_heads = [1,2,4,8],
        l_ratio = [4,4,4,4],
        l_attn_drop = 0.05,
        l_drop_path = 0.1,
        **kwargs
    )

    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]

    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)

@register_model
def low_eff_gcvit_b5(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:
    
    variant = "low_eff_gcvit_b5"

    kwargs.pop('pretrained_cfg', None)

    model_kwargs = dict(
        model_name = "tf_efficientnet_b5.ns_jft_in1k",
        img_size = [384,384],
        use_low = True,
        use_high = False,
        l_dim = 48,
        l_depths = [2,2,6,2],
        l_windows = [12,12,24,12],
        l_heads = [2,4,8,16],
        l_ratio = [3,3,3,3],
        l_attn_drop = 0.1,
        l_drop_path = 0.15,
        **kwargs
    )

    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]

    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)

@register_model
def high_eff_gcvit_b0(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:

    variant = "high_eff_gcvit_b0"

    kwargs.pop('pretrained_cfg', None)

    model_kwargs = dict(
        model_name = "tf_efficientnet_b0.ns_jft_in1k",
        img_size = [224,224],
        use_low = False,
        use_high = True,
        h_dim = 256,
        h_depths = [4],
        h_windows = [7],
        h_heads = [4],
        h_ratio = [4],
        h_drop = 0.05,
        h_drop_path = 0.05,
        **kwargs
    )

    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]

    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)

@register_model
def high_eff_gcvit_b5(pretrained=False, dataset="celeb_df_v2", **kwargs) -> MultiScaleEffGCViT:

    variant = "high_eff_gcvit_b5"

    kwargs.pop('pretrained_cfg', None)

    model_kwargs = dict(
        model_name = "tf_efficientnet_b5.ns_jft_in1k",
        img_size = [384,384],
        use_low = False,
        use_high = True,
        h_dim = 512,
        h_depths = [6],
        h_windows = [12],
        h_heads = [16],
        h_ratio = [3],
        h_drop = 0.1,
        h_drop_path = 0.1,
        **kwargs
    )

    if pretrained:
        pretrained_cfg = _get_config_for_type(variant, dataset)
    else:
        pretrained_cfg = default_cfgs[variant]

    return _create_ms_eff_gcvit(variant,
                              pretrained_cfg=pretrained_cfg,
                              pretrained=pretrained,
                              **model_kwargs)
