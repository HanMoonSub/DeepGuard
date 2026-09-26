import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

# CLIP's own pretraining normalization statistics (not ImageNet's) -
# required since the backbone below is a frozen CLIP vision tower.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class SVDResidualLinear(nn.Module):
    """
    Replaces a pretrained nn.Linear layer with an SVD-decomposed version.

    The weight matrix is split via SVD into a frozen "principal" subspace
    (the largest singular directions, kept as-is to preserve the pretrained
    representation) and a small trainable "residual" subspace (the trailing,
    least significant singular directions). Only the residual subspace is
    updated during fine-tuning.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            residual_dim: int,
            bias: bool,
            init_weight: torch.Tensor,
            ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        U, S, Vh = torch.linalg.svd(init_weight, full_matrices=False)
        r = max(len(S) - residual_dim, 0)

        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        weight_main = U_r @ torch.diag(S_r) @ Vh_r
        self.weight_main = nn.Parameter(weight_main, requires_grad=False)

        U_res, S_res, Vh_res = U[:, r:], S[r:], Vh[r:, :]
        self.U_residual = nn.Parameter(U_res.clone())
        self.S_residual = nn.Parameter(S_res.clone())
        self.V_residual = nn.Parameter(Vh_res.clone())

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_weight = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        weight = self.weight_main + residual_weight
        return F.linear(x, weight, self.bias)


def _replace_self_attn_linears(module: nn.Module, residual_dim: int) -> None:
    for name, child in module.named_children():
        if "self_attn" in name:
            for sub_name, sub_module in list(child.named_modules()):
                if isinstance(sub_module, nn.Linear):
                    parent = child
                    parts = sub_name.split(".")
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    svd_linear = SVDResidualLinear(
                        sub_module.in_features,
                        sub_module.out_features,
                        residual_dim,
                        bias=sub_module.bias is not None,
                        init_weight=sub_module.weight.data.clone(),
                    )
                    if sub_module.bias is not None:
                        svd_linear.bias.data.copy_(sub_module.bias.data)
                    setattr(parent, parts[-1], svd_linear)
        else:
            _replace_self_attn_linears(child, residual_dim)


def apply_svd_residual_to_self_attn(vision_model: nn.Module, residual_dim: int) -> nn.Module:
    """
    Replaces every nn.Linear inside self-attention blocks with SVDResidualLinear,
    then freezes everything in the backbone except the residual (U/S/V) components.
    """
    _replace_self_attn_linears(vision_model, residual_dim)
    for name, param in vision_model.named_parameters():
        param.requires_grad = any(key in name for key in ("U_residual", "S_residual", "V_residual"))
    return vision_model


class Effort(nn.Module):
    """
    Effort (ICML 2025 Oral): "Orthogonal Subspace Decomposition for
    Generalizable AI-Generated Image Detection".

    Wraps a frozen CLIP ViT-L/14 vision backbone, replaces its self-attention
    projection layers with SVDResidualLinear, and trains only the residual
    singular subspace plus a lightweight binary classification head.
    """

    def __init__(
            self,
            clip_model_name: str = "openai/clip-vit-large-patch14",
            svd_residual_dim: int = 1,
            num_classes: int = 1,
            **kwargs,
            ):
        """
        Args:
            clip_model_name: Hugging Face model id for the CLIP vision backbone.
            svd_residual_dim: Number of trailing (least significant) singular
                values kept trainable per self-attention projection layer;
                everything else in the backbone stays frozen.
            num_classes: Number of output classes (1 for a binary real/fake logit).
        """
        super().__init__()
        clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
        # Newer `transformers` releases flatten CLIPVisionModel (embeddings/encoder
        # live directly on it); older releases nest them under `.vision_model`.
        # apply_svd_residual_to_self_attn mutates the passed-in module in place,
        # so this works for either layout without needing to reassign it back.
        vision_backbone = getattr(clip_vision, "vision_model", clip_vision)
        apply_svd_residual_to_self_attn(vision_backbone, residual_dim=svd_residual_dim)
        self.backbone = clip_vision
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden_size, num_classes)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        # Official Effort applies weight decay uniformly to every trainable
        # parameter (including S_residual) via a single flat Adam group, so
        # nothing is excluded here to reproduce that behavior.
        return set()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(pixel_values=x).pooler_output  # (B, hidden_size)
        return self.head(feat)  # (B, num_classes)
