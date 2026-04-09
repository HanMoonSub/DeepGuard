from pydantic import BaseModel
from datetime import datetime

class UserHistory(BaseModel):
    image_id : int
    user_id : int
    image_loc : str
    label : str
    version_type : str
    model_type : str
    domain_type : str
    created_at : datetime

class UserHistory_indi(UserHistory):
    
    score : float
    face_conf : float
    face_ratio : float
    face_brightness : float
    result_msg : str