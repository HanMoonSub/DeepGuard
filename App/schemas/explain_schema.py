from typing import Literal
from pydantic import BaseModel, Field, model_validator

# ── Branch별 허용 기법 ────────────────────────────────────
#
# [LOW branch] 고해상도 feature map → 국소 위조 흔적 포착에 특화
#   - hirescam   : activation * gradient element-wise 곱, 업샘플링 없이 원본 해상도 유지
#                  → 픽셀 단위 경계가 가장 선명 (aug_smooth: O, eigen_smooth: X)
#   - gradcamelementwise : 각 위치별 gradient 부호 고려 후 ReLU → 노이즈 억제, 국소 영역 정밀 (aug_smooth: O, eigen_smooth: O)
#   - layercam   : positive gradient로 activation 공간 가중합산
#                  → 얕은 레이어에서도 안정적, 텍스처/경계 검출에 강함 (aug_smooth: O, eigen_smooth: O)
#
# [HIGH branch] 저해상도 semantic feature map → 전체 구조 파악에 특화
#   - eigengradcam : activation * gradient 의 첫 번째 주성분
#                    → 클래스 판별력 유지하면서 GradCAM보다 부드럽고 EigenCAM보다 정확 (aug_smooth: O, eigen_smooth: X)
#   - gradcamplusplus   : 2차 gradient 활용 → 다중 위조 영역 동시 포착, 작은 영역에 강함 (aug_smooth: O, eigen_smooth: X)
#   - xgradcam     : gradient를 normalized activation으로 스케일링
#                    → attribution 합이 logit 변화량과 수학적으로 일치, 충실도 높음 (aug_smooth: O, eigen_smooth: O)

_LOW_ALLOWED  = {"hirescam", "gradcamelementwise", "layercam"}
_HIGH_ALLOWED = {"eigengradcam", "gradcamplusplus", "xgradcam"}

class ExplainRequest(BaseModel):
    branch_level: Literal["low","high"] = Field("high", 
                                                description="브랜치 레벨. low: 국소 위조 흔적 포착, high: 전역적 위조 흔적 포착")
    explainer_type: str = Field("eigengradcam",
                                description = (
                                    "선택 가능한 XAI 기법. "
                                    "low: [hirescam, gradcamelementwise, layercam], "
                                    "high: [eigengradcam, gradcamplusplus, xgradcam]"
                                ))
    display_type: Literal["heatmap", "contour", "bbox"] = Field("heatmap", 
                                                                description="시각화 형태. heatmap: 전체 분포, contour: 외곽선, bbox: 위조 의심 영역 사각형")
    category: Literal[0, 1] = Field(1, 
                                    description="판단 클래스 인덱스 (0: Real / 1: Fake)")
    overlay_ratio: float = Field(
        0.5, ge=0.0, le=1.0,
        description = "Heatmap 투명도 (0: 히트맵만 강조, 1: 원본 이미지 위주)"
    )
    aug_smooth: bool = Field(
        False,
        description = "TTA(Test Time Augmentation) 적용 여부. 히트맵을 더 객체 중심적으로 정렬"
    )
    eigen_smooth: bool = Field(
        False,
        description = "PCA 기반 노이즈 제거. 지배적인 패턴만 남김"
    )
    
    @model_validator(mode="after")
    def validate_explainer_for_branch(self) -> "ExplainRequest":
        allowed = {"low": _LOW_ALLOWED, "high": _HIGH_ALLOWED}
        if self.explainer_type not in allowed[self.branch_level]:
            raise ValueError(
                f"branch_level='{self.branch_level}'에서 허용된 기법: "
                f"{sorted(allowed[self.branch_level])} "
                f"(입력값: '{self.explainer_type}')"
            )
        return self