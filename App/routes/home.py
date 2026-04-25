from fastapi import APIRouter, Depends
from services import session_svc

router = APIRouter(prefix="/home", tags=["home"])

@router.get("/")
async def home_ui(session_user = Depends(session_svc.get_session_user_opt)):    
    return {
        "session_user": session_user
    }

