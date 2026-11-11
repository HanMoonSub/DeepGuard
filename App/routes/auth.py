from pydantic import BaseModel
from fastapi import APIRouter, Depends, status, Request
from services import auth_svc, session_svc
from sqlalchemy import Connection
from db.database import context_get_conn
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from schemas.auth_schema import RegisterRequest, LoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])

# Check whether the session survives a frontend refresh and return the user info
@router.get("/check", status_code=status.HTTP_200_OK, response_class=JSONResponse, summary= "로그인 상태 확인 API")
async def check_session(session_user = Depends(session_svc.get_session_user_opt)):
    return {"user": session_user}


@router.post("/register", status_code=status.HTTP_201_CREATED,response_class=JSONResponse, summary="회원가입 API")
async def register_user(request: Request,
                        user_info: RegisterRequest,
                        conn: Connection = Depends(context_get_conn)):
    
    # 1. Check for a duplicate email
    user = await auth_svc.get_user_by_email(conn=conn, email=user_info.email)
    if user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 이메일입니다."
        )
    
    # 2. Hash the password
    hashed_password = auth_svc.get_hashed_password(user_info.password)
    
    # 3. Save the new user to the DB
    await auth_svc.register_user(
        conn=conn, 
        name=user_info.name, 
        email=user_info.email,
        hashed_password=hashed_password
    )
    
    return {"message": "회원가입이 성공적으로 완료되었습니다.", "status": "success"}

@router.post("/login", status_code=status.HTTP_200_OK,response_class=JSONResponse ,summary="로그인 API")
async def login_user(request: Request,
                     login_info: LoginRequest,
                     conn: Connection = Depends(context_get_conn)):
    
    # 1. Authenticate the user (DB lookup and password check)
    user = await auth_svc.authenticate_user(conn=conn, email=login_info.email)
    # 2. Handle authentication failure
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="해당 이메일 사용자는 존재하지 않습니다."
        )
    
    is_correct_pw = auth_svc.verify_password(plain_password=login_info.password,
                                    hashed_password=user.hashed_password)
    
    if not is_correct_pw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="등록하신 패스워드와 입력정보가 일치하지 않습니다."
        )

    request.state.session["session_user"] = {"id": user.id, "name": user.name, "email":user.email}


    # 3. Successful login response
    return {
        "message": "로그인에 성공했습니다.",
        "status": "success"
    }

@router.get("/logout", status_code=status.HTTP_200_OK, response_class=JSONResponse, summary="로그아웃")
async def logout_user(request: Request):
    request.state.session.clear()
    return {
        "message": "로그아웃 되었습니다.",
        "status": "success"
    }