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
    """low-level branch와 high-level branch의 CAM을 각각 다른 기법으로 만든 뒤 하나로 결합한다.

    low/high explainer는 같은 모델을 공유하고, 얼굴 크롭도 한 번만 수행한다.
    결합은 모델 입력 해상도의 CAM에서 하므로 CAMEvaluator에도 그대로 넘길 수 있다.

    Args:
        model_name: ms_eff_* 모델 이름. low_/high_/eff_* 같은 단일 branch 모델은 불가.
        dataset: 사전학습 가중치 dataset.
        low: low-level branch에 쓸 CAMExplainer 클래스 (기본값: HiResCAMExplainer).
        high: high-level branch에 쓸 CAMExplainer 클래스 (기본값: EigenGradCAMExplainer).
        fuse: "mean"(low_weight 가중 평균) 또는 "max"(픽셀별 최댓값).
        low_weight: fuse="mean"일 때 low CAM 가중치 (high는 1 - low_weight, 기본값 0.6).
        margin_ratio, conf_thres, model: BaseExplainer와 동일.
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
        # high는 low가 불러온 모델을 그대로 공유 (가중치 중복 로드 방지)
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
        """두 branch의 CAM(각각 0~1 정규화)을 결합하고 다시 0~1로 정규화한다."""
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
        """얼굴 배열로부터 low/high CAM을 각각 만들고 결합한 raw 흑백 CAM을 반환한다."""
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
        """이미지 경로에서 얼굴을 한 번만 탐지/크롭한 뒤 결합 CAM을 만든다."""
        face = self.low_explainer._preprocess_img(img_path)
        return self._build_grayscale_cam_from_array(
            face, category, aug_smooth, eigen_smooth
        )

    def _prepare_masked_cam_from_array(
        self, face: np.ndarray, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """low/high CAM을 branch마다 다른 임계값으로 이진화한 뒤 합친다.

        threshold를 주지 않으면 각 explainer의 branch 고정값(low 0.95, high 0.8)을 쓰고,
        주면 두 branch에 같은 값을 쓴다.
        위조 의심 영역은 두 마스크의 합집합이고, 히트맵 색과 박스 확률은 결합 CAM(low_weight 가중)으로 그린다.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: (결합 cam_recovered, binary_mask, face)
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
        """이미지 경로에서 얼굴을 한 번만 탐지/크롭한 뒤 _prepare_masked_cam_from_array를 수행한다."""
        face = self.low_explainer._preprocess_img(img_path)
        return self._prepare_masked_cam_from_array(face, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name}, dataset={self.dataset}, "
            f"low={self.low_explainer.__class__.__name__}, high={self.high_explainer.__class__.__name__}, "
            f"fuse={self.fuse}, low_weight={self.low_weight}, device={self.device})"
        )
