import torch
import torch.nn as nn
from typing import Optional, Type, List
import timm
from timm.models.layers import DropPath, trunc_normal_
class Mlp(nn.Module):
    """
    Multi-Layer Perceptron Block
    inp: (B, in_dims), out: (B, out_dims)
    """
    
    def __init__(
        self, 
        in_dims: int,
        hidden_dims: Optional[int] = None,
        out_dims: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
    ):
    
        super().__init__()
        hidden_dims = hidden_dims or in_dims
        out_dims = out_dims or in_dims
        
        self.fc1 = nn.Linear(in_dims, hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dims, out_dims)
        
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x
        
class SE(nn.Module):
    """
        Squeeze and excitation block
    """
    
    def __init__(
        self,
        in_chs: int,
        rd_ratio: float = 0.25
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_chs, int(in_chs * rd_ratio), bias=False),
            nn.GELU(),
            nn.Linear(int(in_chs * rd_ratio), in_chs, bias=False),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        B, C, _, _ = x.shape
        
        out = self.pool(x).view(B,C)
        out = self.fc(out).view(B,C,1,1)
        
        return x * out

class ReduceSize(nn.Module):
    """
        inp: (B,H,W,C)
        out: 
            if keep_dim:
                (B,H/2,W/2,C)
            else:
                (B,H/2,W/2,2C)
    """
    def __init__(self,
                 dim: int,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 keep_dim=False):
        super().__init__()
        
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        
        self.reduction = nn.Conv2d(dim, dim_out, kernel_size=3, stride=2,
                                   padding=1, bias=False)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim_out)
        
    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = x.permute(0,3,1,2) # (B,H,W,C) -> (B,C,H,W)
        x = x + self.mb_conv(x) # Skip Connection
        x = self.reduction(x)
        x = x.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
        x = self.norm2(x)
        return (x)
    
class PatchEmbed(nn.Module):
    """
        inp: (B,in_chs,H,W)
        out: (B,H/4,W/4,dim)
    """
    def __init__(self, 
                 in_chs: int = 3, 
                 dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(in_chs, dim, kernel_size=3, stride=2, padding=1)
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0,2,3,1) # (B,C,H,W) -> (B,H,W,C)
        x = self.conv_down(x)
        return x 

def window_partition(x, window_size):
    """
        Args: 
            x: (B,H,W,C)
            window_size
        returns:
            (num_windows * B, window_size, window_size, C)
    """
    B,H,W,C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1,window_size,window_size,C) 
    return windows


def window_reverse(windows, window_size, H, W):
    """
        inp: (B*num_windows, window_size**2, C)
        out: (B, H, W, C)
    """

    C = windows.shape[-1]
    x = windows.view(-1, H//window_size, W//window_size, window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().view(-1,H,W,C)
    return x

class FeatExtract(nn.Module):
    
    """
        Resize: Feature map -> window for global attention
        k번 Repeat(MBConv -> MaxPool 2X2)
        k번: log2(feature map Height // window_size)
    """
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.mb_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.keep_dim = keep_dim
    
    def forward(self, x):
        x = x.contiguous()
        x = x = self.mb_conv(x) # skip connection
        if not self.keep_dim:
            x = self.pool(x)
            
        return x
    
class GlobalQueryGen(nn.Module):
    
    """
        inp: Feature Map(B,D,H,W)
        out: (B,1,num_heads,window_size*window_size,head_dim)
    """
    
    def __init__(self,
                 dim: int,
                 input_resolution: int | float,
                 image_resolution: int | float,
                 window_size: int,
                 num_heads: int):
        super().__init__()
        
        ## Level 0: (B, H//4, W//4, C)
        if input_resolution == image_resolution//4:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )
            
        ## Level 1: (B, H//8, W//8, 2C)
        elif input_resolution == image_resolution//8:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )
        ## Level 2: (B, H//16, W//16, 4C)
        elif input_resolution == image_resolution//16:
            if window_size == input_resolution:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )
            else:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )
        
        ## Level 3: (B, H//32, W//32, 8C)
        elif input_resolution == image_resolution//32:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )
        
        self.resolution = input_resolution
        self.num_heads = num_heads 
        self.N = window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')            
    
    def forward(self, x):
        x = self.to_q_global(x) # (B,C,H,W) -> (B,C,window,window)
        x = x.permute(0,2,3,1) # (B,C,window,window) -> (B,window,window,C)
        x = x.contiguous().view(x.shape[0],1,self.N,self.num_heads,self.dim_head).permute(0,1,3,2,4)
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
                 qk_scale: float | None = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ):

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        
        # Window 토큰 사이의 상대적 h 거리: -(h-1) ~ 0 ~ (h-1)
        # Window 토큰 사이의 상대적 w 거리: -(w-1) ~ 0 ~ (w-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0]) # h의 token 위치 [0,1,2]
        coords_w = torch.arange(self.window_size[1]) # w의 token 위치 [0,1,2]
        # h 좌표: [[0,0,0],[1,1,1],[2,2,2]]
        # w 좌표: [[0,1,2],[0,1,2],[0,1,2]]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # (2, w,w)
        # h: [0,0,0,1,1,1,2,2,2] 
        # w: [0,1,2,0,1,2,0,1,2]
        coords_flatten = torch.flatten(coords, 1) # (2,w*w)
        # window 토큰 사이 거리 구하기: (2, w*w, w*w)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # (w*w, w*w, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1 # 음수 제거, 범위[0~2w-1]
        relative_coords[:, :, 1] += self.window_size[1] - 1 # 음수 제거, 범위[0~2w-1]
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1 # h_idx = h_dix * width
        relative_position_index = relative_coords.sum(-1) # idx = h_idx * width + w_idx, (w*w, w*w)
        
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B, N, C = x.shape # (B*num_windows, window_size**2, dim)
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        # (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, num_heads, N, N)
        
        # ((2W-1)*(2W-1), num_heads) -> (w*w, w*w, num_heads)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # (w*w, w*w, num_hads) -> (num_heads, N, N)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim) -> (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
            
class GlobalWindowAttention(nn.Module):
    """
        inp: (B*num_windows, window_size**2, dim)
        q_query: (B,1,num_heads,window_size**2, head_dim) 
               -> (B, num_windows, num_heads, window_size**2, head_dim)
        out: (B*num_windows,window_size**2, dim)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: int,
                 qkv_bias: bool = True,
                 qk_scale: float | None = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.,
                 ):

        super().__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape # (B_*num_windows, window_size**2, dim)
        B = q_global.shape[0] # B
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        B_dim = torch.div(B_, B, rounding_mode='floor') # num_windows
        # (B_, N, 2, num_heads, head_dim) -> (2, B_, num_heads, N, head_dim)
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2,0,3,1,4)        
        k, v = kv[0], kv[1]
        # (B, num_windows, num_heads, window_size**2, head_dim)
        q_global = q_global.repeat(1, B_dim, 1, 1, 1) # Compute Coefficient
        # (B, num_windows, num_heads, window_size**2, head_dim)
        # (B * num_windows, num_heads, window_size**2, head_dim)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1)) # (B_, num_heads, N, N)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GCViTBlock(nn.Module):
    """
        inp: (B,H,W,C)
        out: (B,H,W,C)
        
        if attention: WindowAttention
            Local Window Attention
        else attention: WindowAttentionGlobal
            Global Window Attention
    """
    def __init__(self,
                dim: int,
                input_resolution: int | float,
                num_heads: int,
                window_size: int = 7,
                mlp_ratio: float = 4.,
                qkv_bias: bool = True,
                qk_scale: Optional[float] = None,
                drop: float = 0.,
                attn_drop: float = 0.,
                drop_path: float = 0.,
                act_layer: Type[nn.Module] = nn.GELU,
                attention=GlobalWindowAttention,
                norm_layer: Type[nn.Module] = nn.LayerNorm,
                layer_scale: int | float | None = None):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        self.attn = attention(
            dim,
            num_heads = num_heads,
            window_size = window_size,
            qkv_bias = qkv_bias,
            qk_scale = qk_scale,
            attn_drop = attn_drop,
            proj_drop = drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_dims=dim, hidden_dims=int(dim*mlp_ratio), act_layer=act_layer, drop=drop)
        
        self.layer_scale = False
        if layer_scale is not None: 
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        
    def forward(self, x, q_global):
        B , H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        # (B*num_windows, window_size, window_size, dim)
        x_windows = window_partition(x, self.window_size) 
        # (B*num_windows, window_size, window_size, dim) -> (B*num_windows, window_size**2, dim)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global) 
        # (B*num_windows, window_size, window_size, dim) -> (B, H, W, dim)
        x = window_reverse(attn_windows, self.window_size, H, W)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        
        return x
    
class GCViTLevel(nn.Module):
    """
        inp: (B,H,W,C)
        out: (B,H//2,W//2,2C)
    """
    
    def __init__(self,
                 dim: int,
                 depth: int,
                 input_resolution: int | float,
                 image_resolution: int | float,
                 num_heads: int,
                 window_size: int,
                 downsample: bool =True,
                 mlp_ratio: float =4.,
                 qkv_bias: bool =True,
                 qk_scale: int | float | None =None,
                 drop: float =0.,
                 attn_drop: float =0.,
                 drop_path: List[float] | float =0.,
                 norm_layer: Type[nn.Module]=nn.LayerNorm,
                 layer_scale: int | float | None =None):
        super().__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(dim=dim,
                       num_heads=num_heads,
                       window_size=window_size,
                       mlp_ratio = mlp_ratio,
                       qkv_bias = qkv_bias,
                       qk_scale = qk_scale,
                       attention=LocalWindowAttention if (i % 2 == 0) else GlobalWindowAttention,
                       drop = drop,
                       attn_drop = attn_drop,
                       drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer = norm_layer,
                       layer_scale = layer_scale,
                       input_resolution=input_resolution)
            for i in range(depth)
            ])
        self.downsample = None if not downsample else ReduceSize(dim=dim, norm_layer=norm_layer)
        self.q_global_gen = GlobalQueryGen(dim, input_resolution, image_resolution, window_size, num_heads)
        
    def forward(self, x):
        q_global = self.q_global_gen(x.permute(0,3,1,2)) #(B,H,W,C) -> (B,C,H,W)
        
        for b in self.blocks:
            x = b(x, q_global)
            
        if self.downsample is None:
            return x
        return self.downsample(x)
    
class GCViT(nn.Module):
    """
        inp: (B,in_chs,H,W)
        out: (B,num_classes)
    """
    
    def __init__(self,
                 dim: int,
                 depths: List[int],
                 window_size: List[int],
                 mlp_ratio: float,
                 num_heads: List[int],
                 image_resolution: int | float,
                 drop_path: float = 0.2,
                 in_chs: int = 3,
                 num_classes: int = 1,
                 qkv_bias: bool = True,
                 qk_scale: int | float | None = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 layer_scale: float = None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            resolution: input image resolution.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        
        super().__init__()
        
        num_feats = int(dim * 2 ** (len(depths)-1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chs=in_chs, dim=dim)
        self.pos_drop = nn.Dropout(drop)
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLevel(
                dim = int(dim * 2 ** i),
                depth = depths[i],
                num_heads = num_heads[i],
                window_size = window_size[i],
                mlp_ratio = mlp_ratio,
                qkv_bias = qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer = norm_layer,
                downsample = (i < len(depths) - 1),
                layer_scale = layer_scale,
                input_resolution = int(2 ** (-2 - i) * image_resolution),
                image_resolution = image_resolution,
            )
            self.levels.append(level)
        
        self.norm = norm_layer(num_feats)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_feats, num_classes) if self.num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_feats(self, x):
        x = self.patch_embed(x) # (B,in_chs,H,W) -> (B,H//4,W//4,dim)
        x = self.pos_drop(x)
            
        for level in self.levels:
            x = level(x)
            
        x = self.norm(x) # (B,H//2**n-1,W//2**n-1,dim*2**n-1)
        x = x.permute(0,3,1,2)# (B,H,W,C) -> (B,C,H,W)
        x = self.avgpool(x) # (B,C,H,W) -> (B,C,1,1)
        x = x.contiguous().view(x.shape[0],-1) # (B,C)
            
        return x
            
    def forward(self, x):
        x = self.forward_feats(x)
        x = self.head(x)
            
        return x