from datetime import datetime
from pydantic import BaseModel, Field


class ImageData(BaseModel):
    """
    [Common meta / history list] Basic identifying info for an image analysis

    Used directly as the list response, and serves as the base for the detail/inference result schemas.

    - routes: image (GET /image/history)
    - services: image_svc.get_user_histories
    - Subclasses: ImageDetailData, ImageResultData
    """
    image_id: int = Field(..., description="이미지 분석 레코드 고유 ID (image_result.id)")
    image_loc: str = Field(..., description="서버 내 저장된 이미지 파일 경로 (DB 저장 형식, 예: /static/uploads/user@a.com/img_1700000000.png)")
    label: str = Field(..., description="딥페이크 판정 결과 라벨 (FAKE / REAL / UNKNOWN)")
    version_type: str = Field(..., description="사용된 모델 버전 (현재 v1만 사용)")
    model_type: str = Field(..., description="모델 속도/정확도 모드 (fast / pro)")
    domain_type: str = Field(..., description="얼굴 도메인 타입 (서양인 / 동양인)")
    created_at: datetime = Field(..., description="분석 요청 생성 시각 (UTC)")


class ImageInference(BaseModel):
    """
    [Inference detail result] Numeric deepfake analysis result

    - Base class only (not returned directly)
    - Subclasses: ImageDetailData, ImageResultData
    - services: image_svc.get_user_history, image_svc.get_image_result
    """
    score: float = Field(..., description="딥페이크 확률 점수 (0.0~1.0, 0.5 이상이면 FAKE 판정). 분석 실패 시 -1.0")
    result_msg: str = Field(..., description="분석 결과에 대한 상세 메시지 (성공/경고/실패 사유)")


class ImageDetailData(ImageData, ImageInference):
    """
    [History detail] Detailed result of a member's individual image analysis

    - routes: image (GET /image/history/{image_id}, DELETE /image/history/{image_id} - internal lookup)
    - services: image_svc.get_user_history
    """
    status: str


class ImageResultData(ImageData, ImageInference):
    """
    [Inference result lookup] Analysis progress status + final result (members and guests)

    - routes: inference (GET /inference/image/{image_id})
    - services: image_svc.get_image_result
    """
    user_id: int | None = Field(None, description="유저 ID. 비회원 분석 요청의 경우 None")
    status: str = Field(..., description="분석 진행 상태 (PENDING / SUCCESS / WARNING / FAILED)", examples=["SUCCESS"])