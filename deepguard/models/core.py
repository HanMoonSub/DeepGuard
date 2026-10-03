import timm
import torch
import torch.nn as nn

# The original CORE (niyunsheng/CORE) normalizes inputs with 0.5 mean/std
# (its default --norm "0.5"), matching the Xception ImageNet weights.
CORE_MEAN = (0.5, 0.5, 0.5)
CORE_STD = (0.5, 0.5, 0.5)


class CORE(nn.Module):
    """
    CORE: Consistent Representation Learning for Face Forgery Detection (CVPRW 2022).

    ImageNet-pretrained Xception, fully trainable. Mirrors the original repo's
    Xception, which returns (feat, logits) with feat = global avg pool of
    ReLU(bn4(conv4(x))) - timm's legacy_xception with num_classes=0 returns
    exactly that 2048-d pooled feature.

    The head is 2-class like the original (trained with CrossEntropy). For
    DeepGuard's sigmoid-based metrics/inference, 'logit' = l_fake - l_real is
    also returned, so sigmoid(logit) == softmax(logits)[:, 1].
    """

    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__()
        self.backbone = timm.create_model('legacy_xception', pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)    # (B, 2048)
        logits = self.head(feat)   # (B, 2)  [real, fake]
        logit = logits[:, 1:] - logits[:, :1]  # (B, 1)

        return {'logit': logit, 'logits': logits, 'feat': feat}
