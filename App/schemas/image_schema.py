from datetime import datetime
from pydantic import BaseModel, Field


class ImageData(BaseModel):
    """
    [공통 메타 / 히스토리 목록] 이미지 분석 기본 식별 정보

    목록 응답으로 직접 사용되며, 상세/추론 결과 스키마의 베이스가 된다.

    - routes: image (GET /image/history)
    - services: image_svc.get_user_histories
    - 상속처: ImageDetailData, ImageResultData
    """
    image_id: int = Field(..., description="이미지 분석 레코드 고유 ID (image_result.id)")
    image_loc: str = Field(..., description="서버 내 저장된 이미지 파일 경로 (DB 저장 형식, 예: /static/uploads/user@a.com/img_1700000000.png)")
    label: str = Field(..., description="딥페이크 판정 결과 라벨 (FAKE / REAL / UNKNOWN)")
    version_type: str = Field(..., description="사용된 모델 버전 (v1 / v2)")
    model_type: str = Field(..., description="모델 속도/정확도 모드 (fast / pro)")
    domain_type: str = Field(..., description="얼굴 도메인 타입 (서양인 / 동양인)")
    created_at: datetime = Field(..., description="분석 요청 생성 시각 (UTC)")


class ImageInference(BaseModel):
    """
    [추론 상세 결과] 딥페이크 분석 수치 결과

    - 상속 베이스 클래스 (직접 응답 X)
    - 상속처: ImageDetailData, ImageResultData
    - services: image_svc.get_user_history, image_svc.get_image_result
    """
    score: float = Field(..., description="딥페이크 확률 점수 (0.0~1.0, 0.5 이상이면 FAKE 판정). 분석 실패 시 -1.0")
    face_conf: float = Field(..., description="얼굴 탐지 신뢰도 (0.0~1.0). 분석 실패 시 -1.0")
    face_ratio: float = Field(..., description="이미지 내 얼굴이 차지하는 면적 비율 (0.0~1.0). 분석 실패 시 -1.0")
    face_brightness: float = Field(..., description="얼굴 영역 평균 밝기 값. 분석 실패 시 -1.0")
    result_msg: str = Field(..., description="분석 결과에 대한 상세 메시지 (성공/경고/실패 사유)")


class ImageDetailData(ImageData, ImageInference):
    """
    [히스토리 상세] 회원 개별 이미지 분석 상세 결과

    - routes: image (GET /image/history/{image_id}, DELETE /image/history/{image_id} - 내부 조회)
    - services: image_svc.get_user_history
    """
    status: str


class ImageResultData(ImageData, ImageInference):
    """
    [추론 결과 조회] 분석 진행 상태 + 최종 결과 (회원/비회원 공통)

    - routes: inference (GET /inference/image/{image_id})
    - services: image_svc.get_image_result
    """
    user_id: int | None = Field(None, description="유저 ID. 비회원 분석 요청의 경우 None")
    status: str = Field(..., description="분석 진행 상태 (PENDING / SUCCESS / WARNING / FAILED)", examples=["SUCCESS"])