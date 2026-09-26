from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# IID's Xception/IResNet branches were both trained/normalized with this
# convention (not ImageNet's) - required for both the trainable implicit
# branch and the frozen explicit branch, since they share the same input.
IID_MEAN = (0.5, 0.5, 0.5)
IID_STD = (0.5, 0.5, 0.5)


# ============================================================================
# Explicit identity extractor: frozen ArcFace-style IResNet-50.
# Layer names match insightface/arcface_torch's iresnet50 so a publicly
# released checkpoint (e.g. ms1mv3_arcface_r50) loads directly via
# load_state_dict.
# ============================================================================

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet50(nn.Module):
    """
    ArcFace-style IResNet-50. Expects 112x112 input (fc_scale assumes a
    7x7 spatial map after four stride-2 stages) and outputs a 512-d
    (by default) face-identity embedding.
    """

    fc_scale = 7 * 7

    def __init__(self, layers=(3, 4, 14, 3), num_features: int = 512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=0.0, inplace=True)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes, eps=1e-05),
            )
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


# ============================================================================
# Implicit identity backbone: trainable Xception, channel-reduced to the
# same embedding size as the explicit extractor.
# ============================================================================

class ImplicitIdentityBackbone(nn.Module):
    def __init__(self, embedding_size: int = 512):
        super().__init__()
        self.xception = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        in_chs = self.xception.feature_info[-1]['num_chs']
        self.adjust_channel = nn.Sequential(
            nn.Conv2d(in_chs, embedding_size, kernel_size=1, stride=1),
            nn.BatchNorm2d(embedding_size),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.xception(x)[-1]
        feat = self.adjust_channel(feat)
        return self.pool(feat).flatten(1)


class IID(nn.Module):
    """
    IID-Net (CVPR 2023): "Implicit Identity Driven Deepfake Face Swapping Detection".

    Classifies the residual between a trainable "implicit identity" embedding
    (Xception, learned on the detection task) and a frozen "explicit identity"
    embedding (ArcFace-style IResNet-50): real faces should have implicit ~
    explicit identity, face-swapped fakes should diverge.

    Only the classification (BCE) and EIC (explicit-implicit consistency)
    losses from the original paper are reproduced (computed in train_iid.py
    from the embeddings this model returns) - the IIE identity-classification
    loss needs per-person identity labels DeepGuard's datasets don't provide,
    so it's intentionally omitted here.
    """

    def __init__(
            self,
            embedding_size: int = 512,
            explicit_extractor_path: Optional[str] = None,
            num_classes: int = 1,
            **kwargs,
            ):
        """
        Args:
            embedding_size: Dimensionality of both the implicit and explicit
                identity embeddings (must match, since they're subtracted).
            explicit_extractor_path: Local path to a pretrained ArcFace-style
                IResNet-50 checkpoint (e.g. insightface's ms1mv3_arcface_r50).
                Required for the explicit branch to be meaningful - without
                it, this extractor is randomly initialized noise and the
                whole explicit-vs-implicit comparison is meaningless.
            num_classes: Number of output classes (1 for a binary real/fake logit).
        """
        super().__init__()
        self.backbone = ImplicitIdentityBackbone(embedding_size)

        self.explicit_extractor = IResNet50(num_features=embedding_size)
        if explicit_extractor_path:
            state_dict = torch.load(explicit_extractor_path, map_location="cpu")
            self.explicit_extractor.load_state_dict(state_dict)
        else:
            print(
                "⚠️  IID: no explicit_extractor_path given - the explicit identity "
                "extractor is randomly initialized. Training will run but the "
                "explicit-vs-implicit identity comparison will be meaningless "
                "until a pretrained ArcFace checkpoint is provided."
            )
        for p in self.explicit_extractor.parameters():
            p.requires_grad = False

        self.head = nn.Linear(embedding_size, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.explicit_extractor.eval()  # always frozen, regardless of model.train()/eval()
        return self

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'bn', 'norm'}

    def forward(self, x: torch.Tensor) -> dict:
        implicit_emb = self.backbone(x)  # (B, embedding_size)

        with torch.no_grad():
            resized = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
            explicit_emb = self.explicit_extractor(resized)  # (B, embedding_size)

        logit = self.head(implicit_emb - explicit_emb)  # (B, num_classes)

        return {
            'logit': logit,
            'implicit_emb': implicit_emb,
            'explicit_emb': explicit_emb,
        }
