from pydantic import BaseModel
from datetime import datetime

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

# 비디오 결과 값 가져오기 
# 사용자 개별 비디오 히스토리 조회
class VideoDataDetail(VideoData):
    
    score : float
    face_conf : float
    face_ratio : float
    face_brightness : float
    result_msg : str

# 비디오 상세 결과 값 가져오기
class VideoFrameData(BaseModel):
    frame_index: int
    frame_time: float
    score: float
    face_conf: float
    face_ratio: float
    face_brightness: float
    
class VideoAnalysisMeta(BaseModel):
    fps: float | None
    total_frames: int | None
    num_sampled: int | None
    num_extracted: int | None
    num_detected: int | None

class VideoDetailResponse(BaseModel):
    meta: VideoAnalysisMeta
    frames: list[VideoFrameData]