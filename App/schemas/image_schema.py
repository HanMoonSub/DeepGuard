from pydantic import BaseModel
from datetime import datetime

class UserHistory(BaseModel):
    image_loc: str
    created_at: datetime