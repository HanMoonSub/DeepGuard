from datetime import datetime

from pydantic import BaseModel, Field


class ArtifactResultData(BaseModel):
    """
    [DeepClip text probabilities] 20 per-region forgery artifact probabilities for FAKE images/videos

    - routes: artifact (GET /artifact/image/{image_id}, GET /artifact/video/{video_id},
              if a stored result already exists, the POST request returns it as is)
    - services: artifact_svc.get_image_artifact, artifact_svc.get_video_artifact
    """
    media_id: int = Field(..., description="image_result.id 또는 video_result.id")
    model_name: str = Field(..., description="사용한 DeepClip 모델 (deepclip_b0 / deepclip_b5)")
    artifact_probs: list[float] = Field(
        ...,
        description="20개 텍스트 확률 (0.0~1.0, PAIR_ORDER 순서, idx = 5 * region + type / "
                    "region: 0 face, 1 eyes, 2 nose, 3 mouth / type: 0 blur, 1 color, 2 texture, 3 structure, 4 blend). "
                    "비디오는 프레임 평균",
    )
    created_at: datetime = Field(..., description="계산 시각")


class ArtifactTaskAccepted(BaseModel):
    """
    [DeepClip text probabilities, async submission] Response for a submitted Celery task when no stored result exists

    - routes: artifact (POST /artifact/image/{image_id}, POST /artifact/video/{video_id})
    """
    message: str = Field(..., description="접수 메세지")
    task_id: str = Field(..., description="결과 조회용 Celery task ID (GET /artifact/result/{task_id})")
