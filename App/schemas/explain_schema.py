from typing import Literal
from pydantic import BaseModel, Field, model_validator

# ── Allowed methods per branch ────────────────────────────────────
#
# [LOW branch] High-resolution feature map → specialized for capturing local forgery artifacts
#   - hirescam   : element-wise activation * gradient, keeps the original resolution without upsampling
#                  → sharpest pixel-level boundaries (aug_smooth: O, eigen_smooth: X)
#   - gradcamelementwise : ReLU after accounting for the gradient sign at each position → suppresses noise, precise on local regions (aug_smooth: O, eigen_smooth: O)
#   - layercam   : spatially weighted sum of activations by positive gradients
#                  → stable even on shallow layers, strong at texture/boundary detection (aug_smooth: O, eigen_smooth: O)
#
# [HIGH branch] Low-resolution semantic feature map → specialized for understanding global structure
#   - eigengradcam : first principal component of activation * gradient
#                    → keeps class discriminability, smoother than GradCAM and more accurate than EigenCAM (aug_smooth: O, eigen_smooth: X)
#   - gradcamplusplus   : uses second-order gradients → captures multiple forged regions at once, strong on small regions (aug_smooth: O, eigen_smooth: X)
#   - xgradcam     : scales gradients by normalized activations
#                    → attributions sum exactly to the logit change, high faithfulness (aug_smooth: O, eigen_smooth: O)
#
# [BOTH] Shared by v1/v2 → combines low (LayerCAM) + high (EigenGradCAM) CAMs with a weighted mean (fuse="mean", low_weight=0.6, fixed on the server)
#
# The forgery threshold is not taken from the client; the fixed per-branch values in explainability are used
#   - high: 0.8, low: 0.95, both: union of the low/high CAMs binarized at 0.95/0.8 respectively
#   - fusion       : v1 BranchFusionExplainer / v2 ClipBranchFusionExplainer (aug_smooth: O, eigen_smooth: X)

_LOW_ALLOWED  = {"hirescam", "gradcamelementwise", "layercam"}
_HIGH_ALLOWED = {"eigengradcam", "gradcamplusplus", "xgradcam"}
_BOTH_ALLOWED = {"fusion"}
_EIGEN_ALLOWED = {"gradcamelementwise", "layercam", "xgradcam"}


class ExplainImageRequest(BaseModel):
    """
    [Image visualization request] CAM visualization parameters for deepfake image forgery artifacts

    - routes: explain (POST /explain/image/{image_id})
    - services: explain_svc.process_explain_image_task
    """
    model_type: Literal["fast", "pro"] = Field("fast",
                                               description="추론 모드 (fast: 속도 우선, pro: 정확도 우선)")
    
    branch_level: Literal["low","high","both"] = Field("high",
                                                description="브랜치 레벨\nlow: 국소 위조 흔적 포착\nhigh: 전역 위조 흔적 포착\nboth: low + high 결합")
    
    explainer_type: str = Field("eigengradcam", 
                                description = ("선택 가능한 XAI 기법. low: [hirescam, gradcamelementwise, layercam], ""high: [eigengradcam, gradcamplusplus, xgradcam], both: [fusion]"))
    
    display_type: Literal["heatmap", "bbox", "heatmap_bbox"] = Field("heatmap_bbox", 
                                                                     description="시각화 형태. heatmap: 전체 분포, bbox: 위조 의심 영역 사각형, heatmap_bbox: 위조 의심 사각형 내부에 블러 처리된 히트맵을 중첩")
    
    overlay_ratio: float = Field(0.8, ge=0.5, le=1.0, 
                                 description = "Heatmap 투명도 (0: 히트맵만 강조, 1: 원본 이미지 위주)")
    
    aug_smooth: bool = Field(False, 
                             description = "TTA(Test Time Augmentation) 적용 여부. 히트맵을 더 객체 중심적으로 정렬")
    
    eigen_smooth: bool = Field(False, 
                               description = "PCA 기반 노이즈 제거. 지배적인 패턴만 남김")
    
    @model_validator(mode="after")
    def validate_explainer_for_branch(self) -> "ExplainImageRequest":
        branch_allowed = {"low": _LOW_ALLOWED, "high": _HIGH_ALLOWED, "both": _BOTH_ALLOWED}
        if self.explainer_type not in branch_allowed[self.branch_level]:
            raise ValueError(
                f"branch_level='{self.branch_level}'에서 허용된 기법: "
                f"{sorted(branch_allowed[self.branch_level])} "
                f"(입력값: '{self.explainer_type}')"
            )
        return self

    @model_validator(mode="after")
    def validate_explainer_for_eigen_smooth(self) -> "ExplainImageRequest":
        if self.eigen_smooth and self.explainer_type not in _EIGEN_ALLOWED:
            raise ValueError(
                f"eigen_smooth가 허용된 기법: "
                f"{sorted(_EIGEN_ALLOWED)} "
                f"(입력값: '{self.explainer_type}')"
            )
        return self
    
class ExplainFrameRequest(ExplainImageRequest):
    """
    [Frame visualization request] CAM visualization parameters for deepfake video frame forgery artifacts

    Uses the same parameters as ExplainImageRequest, applied to visualizing a specific video frame.

    - routes: explain (POST /explain/video/{video_id}/frame/{frame_index})
    - services: explain_svc.process_explain_frame_task
    """
    pass