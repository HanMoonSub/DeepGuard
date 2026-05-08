from pydantic import BaseModel
from datetime import datetime

# [히스토리 목록] GET /video/history
# 사용자 전체 비디오 히스토리 조회
class VideoData(BaseModel):
    id: int
    user_id : int | None
    video_loc : str
    status : str
    label : str
    version_type : str
    model_type : str
    domain_type : str
    created_at : datetime

# [히스토리 상세 / 추론 결과] GET /video/history/{video_id}, GET /inference/video/{video_id}
# 개별 비디오 상세 조회 + 추론 결과값 포함
class VideoDetailData(VideoData):
    
    score : float
    face_conf : float
    face_ratio : float
    face_brightness : float
    result_msg : str

# [프레임별 추론 결과] GET /inference/video/{video_id}/detail 응답 내부
# video_frame_result 테이블 row 단위 매핑
class VideoFrameData(BaseModel):
    frame_index: int
    frame_time: float
    score: float
    face_conf: float
    face_ratio: float
    face_brightness: float
    
# [비디오 메타 정보] GET /inference/video/{video_id}/detail 응답 내부
# video_meta_result 테이블 row 단위 매핑 
class VideoMetaData(BaseModel):
    fps: float 
    total_frames: int # 전체 프레임 수 
    num_sampled: int # 샘플링 대상 프레임 수 
    num_extracted: int # 실제 추출된 프레임 수 
    num_detected: int # 얼굴 탐지 성공 프레임 수

# [상세 분석 최종 응답] GET /inference/video/{video_id}/{detail}
# 로그인 유저 전용 (get_session_user_prt)
class VideoDetailResponse(BaseModel):
    meta: VideoMetaData
    frames: list[VideoFrameData]