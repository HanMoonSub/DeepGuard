from fastapi import APIRouter, Depends, status, Request
from fastapi.exceptions import HTTPException

def get_session_user_opt(request:Request):
    if "session_user" in request.state.session:
        return request.state.session["session_user"]
    

def get_session_user_prt(request:Request):
    if "session_user" not in request.state.session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="해당 서비스는 로그인이 필요합니다.")
    
    return request.state.session["session_user"]