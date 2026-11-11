from typing import Literal

import numpy as np
import torch

from explainability.explainer.cam_explainer import (
    CAMDisplayMixin,
    CAMExplainer,
    EigenGradCAMExplainer,
    HiResCAMExplainer,
)


class BranchFusionExplainer(CAMDisplayMixin):
    """Build CAMs for the low-level and high-level branches with different methods, then combine them into one.

    The low/high explainers share the same model, and the face is cropped only once.
    Combining happens on CAMs at the model input resolution, so the result can be passed straight to CAMEvaluator.

    Args:
        model_name: ms_eff_* model name. Single-branch models such as low_/high_/eff_* are not allowed.
        dataset: pretrained weight dataset.
        low: CAMExplainer class for the low-level branch (default: HiResCAMExplainer).
        high: CAMExplainer class for the high-level branch (default: EigenGradCAMExplainer).
        fuse: "mean" (weighted average by low_weight) or "max" (per-pixel maximum).
        low_weight: weight of the low CAM when fuse="mean" (high gets 1 - low_weight, default 0.6).
        margin_ratio, conf_thres, model: same as BaseExplainer.
    """

    def __init__(
        self,
        model_name: Literal[
            "ms_eff_vit_b0", "ms_eff_vit_b5", "ms_eff_gcvit_b0", "ms_eff_gcvit_b5"
        ],
        dataset: Literal["celeb_df_v2", "ff++", "kodf"] | None = None,
        low: type[CAMExplainer] = HiResCAMExplainer,
        high: type[CAMExplainer] = EigenGradCAMExplainer,
        fuse: Literal["mean", "max"] = "mean",
        low_weight: float = 0.6,
        margin_ratio: float = 0.2,
        conf_thres: float = 0.5,
        model: torch.nn.Module | None = None,
    ):
        self._validate_explainer_cls(low)
        self._validate_explainer_cls(high)
        if fuse not in ("mean", "max"):
            raise ValueError(f"fuse는 'mean' 또는 'max'만 가능합니다 (입력: {fuse!r}).")
        if not 0.0 <= low_weight <= 1.0:
            raise ValueError(f"low_weight는 0~1 사이여야 합니다 (입력: {low_weight}).")

        self.model_name = model_name
        self.dataset = dataset
        self.fuse = fuse
        self.low_weight = low_weight

        common = {
            "model_name": model_name,
            "dataset": dataset,
            "margin_ratio": margin_ratio,
            "conf_thres": conf_thres,
        }
        self.low_explainer = low(branch_level="low", model=model, **common)
        # high reuses the model loaded by low as-is (avoids loading the weights twice)
        self.high_explainer = high(
            branch_level="high", model=self.low_explainer.model, **common
        )

        self.model = self.low_explainer.model
        self.device = self.low_explainer.device

    def _validate_explainer_cls(self, cls: type) -> None:
        if not (isinstance(cls, type) and issubclass(cls, CAMExplainer)):
            raise TypeError(
                f"low/high에는 CAMExplainer 하위 클래스를 넘겨야 합니다 (입력: {cls!r})."
            )

    def _fuse_cams(self, low_cam: np.ndarray, high_cam: np.ndarray) -> np.ndarray:
        """Combine the two branch CAMs (each normalized to 0~1) and normalize the result to 0~1 again."""
        if self.fuse == "max":
            fused = np.maximum(low_cam, high_cam)
        else:
            fused = self.low_weight * low_cam + (1.0 - self.low_weight) * high_cam
        fused = fused - fused.min()
        return (fused / (fused.max() + 1e-7)).astype(np.float32)

    def _build_grayscale_cam_from_array(
        self,
        face: np.ndarray,
        category: int = 1,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """Build the low/high CAMs from a face array and return the combined raw grayscale CAM."""
        low_cam, tensor, face = self.low_explainer._build_grayscale_cam_from_array(
            face, category, aug_smooth, eigen_smooth
        )
        high_cam, _, _ = self.high_explainer._build_grayscale_cam_from_array(
            face, category, aug_smooth, eigen_smooth
        )
        return self._fuse_cams(low_cam, high_cam), tensor, face

    def _build_grayscale_cam(
        self,
        img_path: str,
        category: int = 1,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        """Detect/crop the face from the image path only once, then build the combined CAM."""
        face = self.low_explainer._preprocess_img(img_path)
        return self._build_grayscale_cam_from_array(
            face, category, aug_smooth, eigen_smooth
        )

    def _prepare_masked_cam_from_array(
        self, face: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Binarize the low/high CAMs with a different threshold per branch, then merge them.

        If threshold is not given, each explainer's fixed branch value (low 0.95, high 0.8) is used;
        if given, the same value is used for both branches.
        The suspected forgery region is the union of the two masks; heatmap colors and box probabilities come from the combined CAM (weighted by low_weight).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (combined cam_recovered, binary_mask, face)
        """
        low_cam, low_mask, face = self.low_explainer._prepare_masked_cam_from_array(
            face, **kwargs
        )
        high_cam, high_mask, _ = self.high_explainer._prepare_masked_cam_from_array(
            face, **kwargs
        )
        return self._fuse_cams(low_cam, high_cam), np.maximum(low_mask, high_mask), face

    def _prepare_masked_cam(
        self, img_path: str, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect/crop the face from the image path only once, then run _prepare_masked_cam_from_array."""
        face = self.low_explainer._preprocess_img(img_path)
        return self._prepare_masked_cam_from_array(face, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, dataset={self.dataset}, "
            f"low={self.low_explainer.__class__.__name__}, high={self.high_explainer.__class__.__name__}, "
            f"fuse={self.fuse}, low_weight={self.low_weight}, device={self.device})"
        )
