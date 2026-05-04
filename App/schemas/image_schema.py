from pydantic import BaseModel
from datetime import datetime

class BaseMetadata(BaseModel):
    image_id: int
    image_loc: str
    label: str
    version_type: str
    model_type: str
    domain_type: str
    created_at: datetime
class InferenceResult(BaseModel):
    score : float
    face_conf : float
    face_ratio : float
    face_brightness : float
    result_msg : str
class UserHistory(BaseMetadata):
    user_id: int

class UserHistory_indi(UserHistory, InferenceResult):   
    pass
class ImageData_indi(BaseMetadata, InferenceResult):
    user_id : int | None
    status : str
