from typing import Literal

import timm
import torch

from explainability.explainer.cam_explainer import (
    EigenGradCAMExplainer,
    GradCAMElementWiseExplainer,
    GradCAMPlusPlusExplainer,
    HiResCAMExplainer,
    LayerCAMExplainer,
    XGradCAMExplainer,
)
from explainability.utils.deepclip import check_deepclip_model_name


class ClipExplainerMixin:
    """
    DeepClip parts shared by the Clip explainers.

    DeepClip has no dataset variants, and its low/high GCViT branches live under model.image_encoder.
    Both branches are always used, so branch_level can be "low", "high" or "both".
    forward() returns (B,1) fake logits, same as DeepGuard, so the CAM targets are unchanged.
    """

    def __init__(self, model_name: Literal["deepclip_b0", "deepclip_b5"], **kwargs):
        check_deepclip_model_name(model_name)
        kwargs.pop("dataset", None)
        super().__init__(model_name=model_name, dataset=None, **kwargs)

    def _load_model(
        self, model_name: str, dataset: str | None = None
    ) -> torch.nn.Module:
        return timm.create_model(model_name, pretrained=True)

    def _get_transformer_type(self) -> Literal["gcvit"]:
        return "gcvit"

    def _branch_root(self) -> torch.nn.Module:
        return self.model.image_encoder


# low-level branch
class ClipHiResCAMExplainer(ClipExplainerMixin, HiResCAMExplainer):
    pass


class ClipGradCAMElementWiseExplainer(ClipExplainerMixin, GradCAMElementWiseExplainer):
    pass


class ClipLayerCAMExplainer(ClipExplainerMixin, LayerCAMExplainer):
    pass


# high-level branch
class ClipEigenGradCAMExplainer(ClipExplainerMixin, EigenGradCAMExplainer):
    pass


class ClipGradCAMPlusPlusExplainer(ClipExplainerMixin, GradCAMPlusPlusExplainer):
    pass


class ClipXGradCAMExplainer(ClipExplainerMixin, XGradCAMExplainer):
    pass
