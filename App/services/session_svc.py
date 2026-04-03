from fastapi import APIRouter, Depends, status, Request
from fastapi.exceptions import HTTPException

def get_session_user_opt(request:Request):
    if "session_user" in request.session:
        return request.session["session_user"]
    

def get_session_user_prt(request:Request):
    if "session_user" not in request.session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="해당 서비스는 로그인이 필요합니다.")
    
    return request.session["session_user"]