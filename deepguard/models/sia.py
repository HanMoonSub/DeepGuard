import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# DeepfakeBench's sia.yaml normalizes inputs with 0.5 mean/std.
SIA_MEAN = (0.5, 0.5, 0.5)
SIA_STD = (0.5, 0.5, 0.5)


class SAIA_conv(nn.Module):
    """
    Self-information based spatial/channel attention (port of DeepfakeBench's SAIA_conv).

    No trainable parameters: the distance statistics come from fixed all-ones
    depthwise kernels and are computed without gradients; only the attended
    feature map carries gradients back to the backbone. Statistics are computed
    in fp32 (autocast off) so AMP can't overflow / divide by a zero fp16 mean.
    """

    def __init__(self, outdim, kernel_size=3, padding=1, isspace=True, ischannel=True):
        super().__init__()

        self.band_width = 1.0

        self.isspace = isspace
        self.ischannel = ischannel
        self.outdim = outdim

        self.register_buffer("weight", torch.ones((outdim, 1, kernel_size, kernel_size)), persistent=False)
        self.register_buffer("weight2", torch.ones((outdim, 1, 1, 1)) * (kernel_size * kernel_size), persistent=False)
        self.pad = padding
        self.channel_range = 5

    def forward(self, x):
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            batch_size, num_channel = xf.shape[:2]

            # intra-feature
            x1 = F.conv2d(xf, self.weight, padding=self.pad, groups=self.outdim)
            x2 = F.conv2d(xf, self.weight2, padding=0, groups=self.outdim)
            intra_distance = torch.abs(x2 - x1)

            # inter-feature
            pad_x = torch.cat([xf, xf[:, :self.channel_range + 1, :, :]], dim=1)
            distances = []
            for i in range(1, self.channel_range + 1):
                distances.append(xf - pad_x[:, i:num_channel + i, :, :])

            distance = torch.cat(distances, dim=1)
            _, _, h_dis, w_dis = distance.shape
            distance = distance.view(batch_size, -1, self.channel_range, h_dis, w_dis).sum(dim=2)
            inter_distance = torch.abs(distance.view(batch_size, -1, h_dis, w_dis))
            att = intra_distance + 0.5 * inter_distance

            if self.ischannel:
                # using mean of distance to normalize
                distance_channel = torch.exp(-att / att.mean() / 2 / self.band_width ** 2)
                distance_channel = -torch.log(distance_channel + 0.1)
                channel_attention = torch.mean(distance_channel.view(batch_size, self.outdim, -1), dim=2)
                channel_attention = (channel_attention.view(batch_size, -1, 1, 1) + 1).to(x.dtype)

            if self.isspace:
                space_attention = (att / att.mean() / 2 / self.band_width ** 2)
                space_scale = (torch.sigmoid(space_attention) + 1).to(x.dtype)
                space_attention = space_attention.to(x.dtype)

        if self.isspace and self.ischannel:
            return space_scale * x * channel_attention.expand_as(x), space_attention
        elif self.isspace:
            return space_scale * x, x
        elif self.ischannel:
            return x * channel_attention.expand_as(x), x
        return x, x


def _conv_bn_relu(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, 1, 1, 0),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(inplace=True),
    )


class SIA(nn.Module):
    """
    SIA: An Information Theoretic Approach for Attention-Driven Face Forgery Detection (ECCV 2022).

    Follows DeepfakeBench's SIADetector (no official code is public):
    ImageNet EfficientNet-B4 with SAIA attention after stages 2/3/5 (32/56/160 ch)
    and cross-stage residuals, 2-class head on the pooled 1792-d feature.

    timm's tf_efficientnet_b4 has the same 32 blocks as efficientnet_pytorch's
    efficientnet-b4, grouped into 7 stages [2,4,4,6,6,8,2], so the end of
    blocks[1] / blocks[2] / blocks[4] == DeepfakeBench's _blocks[5] / [9] / [21].

    Like DeepfakeBench, the pretrained stem conv is replaced by a freshly
    initialized nn.Conv2d(3, 48, 3, stride=2) (padding 0).

    'logit' = l_fake - l_real is also returned, so sigmoid(logit) == softmax(logits)[:, 1].
    """

    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__()
        self.backbone = timm.create_model('tf_efficientnet_b4.aa_in1k', pretrained=True, num_classes=0)
        self.backbone.conv_stem = nn.Conv2d(3, 48, kernel_size=3, stride=2, bias=False)

        self.att1conv = SAIA_conv(32, kernel_size=3, isspace=True, ischannel=True)
        self.att2conv = SAIA_conv(56, kernel_size=3, isspace=True, ischannel=True)
        self.att4conv = SAIA_conv(160, kernel_size=3, isspace=True, ischannel=True)

        self.conv1 = _conv_bn_relu(32, 56)
        self.conv2 = _conv_bn_relu(32, 160)
        self.conv3 = _conv_bn_relu(56, 160)

        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone

        # Stem
        x = b.bn1(b.conv_stem(x)) # BatchNormAct2d (BN + SiLU)

        x = b.blocks[1](b.blocks[0](x)) # _blocks[0..5], 32 ch

        x, att1 = self.att1conv(x)
        res1 = self.conv1(att1)
        res2 = self.conv2(att1)

        x = b.blocks[2](x) # _blocks[6..9], 56 ch
        res12 = F.adaptive_max_pool2d(res1, x.shape[-2:]) # == AdaptiveMaxPool2d((32,32)) at 256

        x, att2 = self.att2conv(x + res12)
        res24 = self.conv3(att2)

        x = b.blocks[4](b.blocks[3](x)) # _blocks[10..21], 160 ch
        res14 = F.adaptive_max_pool2d(res2, x.shape[-2:]) # == AdaptiveMaxPool2d((16,16)) at 256
        res24 = F.adaptive_max_pool2d(res24, x.shape[-2:])

        x, _ = self.att4conv(x + res24 + res14)

        x = b.blocks[6](b.blocks[5](x)) # _blocks[22..31]

        # Head
        x = b.bn2(b.conv_head(x)) # BatchNormAct2d (BN + SiLU)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        feat = F.adaptive_avg_pool2d(x, 1).flatten(1) # (B, 1792)
        logits = self.head(feat) # (B, 2)  [real, fake]
        logit = logits[:, 1:] - logits[:, :1] # (B, 1)

        return {'logit': logit, 'logits': logits, 'feat': feat}
