from fastapi import APIRouter, Depends, status
from services import session_svc
from sqlalchemy import Connection
from db.database import context_get_conn


router = APIRouter(prefix="/inference", tags=["inference"])

# ---- 딥페이크 비동기 이미지 추론 API ------
@router.post("image", status_code=status.HTTP_200_OK)
async def predict_image(conn: Connection = Depends(context_get_conn),
                        session_user = Depends(session_svc.get_session_user_opt)):
    return None