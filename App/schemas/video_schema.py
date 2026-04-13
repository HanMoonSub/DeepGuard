from pydantic import BaseModel
from datetime import datetime

class VideoData(BaseModel):
    id: int
    user_id : int | None
    video_loc : str
    status : str
    label : str
    score: float
    face_conf: float
    face_ratio: float
    face_brightness: float
    version_type : str
    model_type : str
    domain_type : str
    result_msg: str
    created_at : datetime