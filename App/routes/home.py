from fastapi import APIRouter, Depends, status, Request
from services import session_svc

router = APIRouter(prefix="/home", tags=["home"])

@router.get("/")
async def home_ui(request: Request,
                  session_user = Depends(session_svc.get_session_user_opt)):
    print(session_user)
    
    return {
        "session_user": session_user
    }

