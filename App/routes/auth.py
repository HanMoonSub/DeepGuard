from fastapi import APIRouter, Depends, status
from services import auth_svc
from sqlalchemy import Connection
from db.database import context_get_conn
from fastapi.exceptions import HTTPException
from schemas.auth_schema import RegisterRequest, LoginRequest

router = APIRouter(prefix="/auth", tags=["auth"])

# --- 회원가입 API ---
@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(req: RegisterRequest,
                        conn: Connection = Depends(context_get_conn)):
    
    # 1. 이메일 중복 확인
    user = await auth_svc.get_user_by_email(conn=conn, email=req.email)
    if user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 이메일입니다."
        )
    
    # 2. 비밀번호 암호화 (해싱)
    hashed_password = auth_svc.get_hashed_password(req.password)
    
    # 3. 신규 유저 DB 저장
    await auth_svc.register_user(
        conn=conn, 
        name=req.name, 
        email=req.email,
        hashed_password=hashed_password
    )
    
    return {"message": "회원가입이 성공적으로 완료되었습니다.", "status": "success"}


# --- 로그인 API ---
@router.post("/login", status_code=status.HTTP_200_OK)
async def login_user(req: LoginRequest,
                     conn: Connection = Depends(context_get_conn)):
    
    # 1. 유저 인증 (DB 조회 및 비밀번호 대조)
    user = await auth_svc.authenticate_user(conn=conn, email=req.email, password=req.password)
    
    # 2. 인증 실패 처리
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 일치하지 않습니다."
        )
    
    # 3. 로그인 성공 응답
    return {
        "message": "로그인에 성공했습니다.",
        "status": "success"
    }