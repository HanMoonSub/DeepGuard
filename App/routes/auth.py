from fastapi import APIRouter,Form, Depends, status
from services import auth_svc
from pydantic import EmailStr
from sqlalchemy import Connection
from db.database import context_get_conn
from fastapi.exceptions import HTTPException


router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(name: str = Form(min_length=5, max_length=100),
                        email: EmailStr = Form(...),
                        password: str = Form(min_length=5, max_length=30),
                        conn: Connection = Depends(context_get_conn)):
    
    # 1. 중복 가입 방지: 입력된 이메일이 이미 DB에 존재하는지 확인합니다.
    user = await auth_svc.get_user_by_email(conn=conn, email=email)
    if user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 이메일입니다."
        )
    
    # 2. 보안 처리: 사용자의 비밀번호를 평문으로 저장하지 않고, 편향된 해시(Hash) 알고리즘으로 암호화합니다.
    hashed_password = auth_svc.get_hashed_password(password)
    
    # 3. 데이터 영속화: 암호화된 비밀번호와 함께 최종 회원 정보를 데이터베이스(User Table)에 기록합니다.
    await auth_svc.register_user(
        conn=conn, 
        name=name, 
        email=email,
        hashed_password=hashed_password
    )
    
    return {"message": "회원가입이 성공적으로 완료되었습니다."}