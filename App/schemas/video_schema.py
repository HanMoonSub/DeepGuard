from datetime import datetime
from pydantic import BaseModel, Field

class VideoData(BaseModel):
    """
    [히스토리 목록] 사용자 전체 비디오 히스토리 조회
    
    - routes: video (GET /video/history)
    - services: video_svc.get_user_histories
    """
    id: int = Field(..., description="비디오 분석 레코드 고유 ID (video_result.id)")
    user_id: int | None = Field(None, description="분석 요청 유저 ID. 비회원의 경우 None")
    video_loc: str = Field(..., description="서버 내 저장된 비디오 파일 경로 (예: /static/uploads/user@a.com/vid_1700000000.mp4)")
    status: str = Field(..., description="분석 진행 상태 (PENDING / SUCCESS / WARNING / FAILED)")
    label: str = Field(..., description="딥페이크 판정 결과 라벨 (FAKE / REAL / UNKNOWN)")
    version_type: str = Field(..., description="사용된 모델 버전 (v1 / v2)")
    model_type: str = Field(..., description="모델 속도/정확도 모드 (fast / pro)")
    domain_type: str = Field(..., description="얼굴 도메인 타입 (서양인 / 동양인)")
    created_at: datetime = Field(..., description="분석 요청 생성 시각 (UTC)")

class VideoDetailData(VideoData):
    """
    [히스토리 상세 / 추론 결과] 개별 비디오 상세 + 추론 수치 결과
    
    - routes: video (GET /video/history/{video_id}), inference (GET /inference/video/{video_id})
    - services: video_svc.get_user_history, video_svc.get_video_result
    """
    score: float = Field(..., description="비디오 전체 평균 딥페이크 확률 (0.0~1.0). 실패 시 -1.0")
    face_conf: float = Field(..., description="평균 얼굴 탐지 신뢰도. 실패 시 -1.0")
    face_ratio: float = Field(..., description="평균 얼굴 면적 비율. 실패 시 -1.0")
    face_brightness: float = Field(..., description="평균 얼굴 영역 밝기. 실패 시 -1.0")
    result_msg: str = Field(..., description="분석 결과 상세 메시지")

class VideoFrameData(BaseModel):
    """
    [프레임별 추론 결과] 비디오 프레임 단위 분석 결과
    
    - routes: inference (GET /inference/video/{video_id}/detail 응답 내부)
    - services: video_svc.get_video_frame_result, save_video_frame_result
    """
    frame_index: int = Field(..., description="비디오 내 프레임 인덱스 (0부터 시작)")
    frame_time: float = Field(..., description="해당 프레임의 영상 내 재생 시점 (초 단위)")
    score: float = Field(..., description="해당 프레임의 딥페이크 확률 점수 (0.0~1.0)")
    face_conf: float = Field(..., description="해당 프레임 얼굴 탐지 신뢰도")
    face_ratio: float = Field(..., description="해당 프레임 얼굴 면적 비율")
    face_brightness: float = Field(..., description="해당 프레임 얼굴 영역 밝기")

class VideoMetaData(BaseModel):
    """
    [비디오 메타 정보] 비디오 전체 프레임 처리 통계
    
    - routes: inference (GET /inference/video/{video_id}/detail 응답 내부)
    - services: video_svc.get_video_meta_result, save_video_meta_result
    """
    fps: float = Field(..., description="비디오의 초당 프레임 수 (Frames Per Second)")
    total_frames: int = Field(..., description="비디오 전체 프레임 수")
    num_sampled: int = Field(..., description="추론을 위해 샘플링한 프레임 수")
    num_extracted: int = Field(..., description="실제로 추출에 성공한 프레임 수")
    num_detected: int = Field(..., description="얼굴 탐지에 성공한 프레임 수 (score 산출 성공)")

class VideoDetailResponse(BaseModel):
    """
    [상세 분석 최종 응답] 비디오 상세 분석 화면용 통합 응답
    
    사용처:
    - routes: inference (GET /inference/video/{video_id}/detail) - 로그인 유저 전용
    """
    meta: VideoMetaData = Field(..., description="비디오 메타 정보 (FPS, 프레임 수 등)")
    frames: list[VideoFrameData] = Field(..., description="프레임별 상세 분석 결과 리스트 (frame_index 오름차순)")