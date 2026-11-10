from typing import Literal

import timm
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from explainability.explainer.cam_explainer import (
    EigenGradCAMExplainer,
    GradCAMElementWiseExplainer,
    GradCAMPlusPlusExplainer,
    HiResCAMExplainer,
    LayerCAMExplainer,
    XGradCAMExplainer,
)
from explainability.explainer.fusion_explainer import BranchFusionExplainer
from explainability.utils.deepclip import check_deepclip_model_name
from explainability.utils.model_targets import BinaryClassifierColumnTarget


class DeepClipCAMModel(torch.nn.Module):
    """
    CAM용 DeepClip 래퍼: forward가 fake logit과 20개 artifact logit을 함께 반환한다.

        (B,3,H,W) -> (B,21)
            column 0      : fake logit (DeepClip.forward와 같음)
            column 1 + i  : PAIR_ORDER[i] (region, artifact type)의 logit
    """

    def __init__(self, clip: torch.nn.Module):
        super().__init__()
        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_emb = self.clip.encode_image(x)
        fake_logit = self.clip.real_fake_logits(img_emb).unsqueeze(-1)
        return torch.cat([fake_logit, self.clip.pair_logits(img_emb)], dim=-1)


class ClipExplainerMixin:
    """
    DeepClip parts shared by the Clip explainers.

    DeepClip has no dataset variants, and its low/high GCViT branches live under model.image_encoder.
    branch_level is "low" or "high"; use ClipBranchFusionExplainer to combine both branches.
    The model is wrapped in DeepClipCAMModel ((B,21) = fake logit + 20 artifact logits), so the CAM can target
    either the fake logit (category) or one artifact (artifact_idx, index in PAIR_ORDER).
    """

    def __init__(self, model_name: Literal["deepclip_b0", "deepclip_b5"], **kwargs):
        check_deepclip_model_name(model_name)
        kwargs.pop("dataset", None)
        model = kwargs.get("model")
        if model is not None and not isinstance(model, DeepClipCAMModel):
            kwargs["model"] = DeepClipCAMModel(model)
        super().__init__(model_name=model_name, dataset=None, **kwargs)

    def _load_model(
        self, model_name: str, dataset: str | None = None
    ) -> torch.nn.Module:
        return DeepClipCAMModel(timm.create_model(model_name, pretrained=True))

    def _get_transformer_type(self) -> Literal["gcvit"]:
        return "gcvit"

    def _branch_root(self) -> torch.nn.Module:
        return self.model.clip.image_encoder

    def _make_targets(self, category: int, artifact_idx: int | None = None) -> list:
        """artifact_idx가 있으면 해당 artifact logit(1 + artifact_idx 열), 없으면 fake logit(0열)을 타겟으로 한다."""
        if artifact_idx is not None:
            if not 0 <= artifact_idx < 20:
                raise ValueError(f"artifact_idx는 0~19 사이여야 합니다 (입력: {artifact_idx}).")
            return [ClassifierOutputTarget(1 + artifact_idx)]
        return [BinaryClassifierColumnTarget(category, column=0)]


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


class ClipBranchFusionExplainer(BranchFusionExplainer):
    """BranchFusionExplainer의 DeepClip 버전: dataset 없음, low/high에는 Clip*Explainer 클래스만 허용."""

    def __init__(
        self,
        model_name: Literal["deepclip_b0", "deepclip_b5"],
        low: type[ClipExplainerMixin] = ClipHiResCAMExplainer,
        high: type[ClipExplainerMixin] = ClipEigenGradCAMExplainer,
        **kwargs,
    ):
        kwargs.pop("dataset", None)
        super().__init__(
            model_name=model_name, dataset=None, low=low, high=high, **kwargs
        )

    def _validate_explainer_cls(self, cls: type) -> None:
        super()._validate_explainer_cls(cls)
        if not issubclass(cls, ClipExplainerMixin):
            raise TypeError(
                f"ClipBranchFusionExplainer에는 Clip*Explainer 클래스를 넘겨야 합니다 (입력: {cls.__name__})."
            )
