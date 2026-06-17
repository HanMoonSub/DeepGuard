from fastapi import APIRouter, Depends, status
from services import session_svc
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/home", tags=["home"])

@router.get("", status_code=status.HTTP_200_OK,response_class=JSONResponse, summary="세션 유저 정보 가져오기")
async def home_ui(session_user = Depends(session_svc.get_session_user_opt)):    
    return {
        "session_user": session_user
    }

