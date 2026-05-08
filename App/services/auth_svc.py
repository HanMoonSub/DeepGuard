from datetime import datetime, timedelta, timezone
from fastapi import status
from fastapi.exceptions import HTTPException
from passlib.context import CryptContext
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from schemas.auth_schema import UserData, UserDataPASS

# 비밀번호 암호화 및 검증
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_hashed_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

# 회원가입 시, 이미 존재하는 이메일과 중복 확인
async def get_user_by_email(conn: Connection, email: str):
    try:
        query = f"""
        SELECT id, name, email from user
        where email = :email
        """

        stmt = text(query)
        bind_stmt = stmt.bindparams(email=email)
        result = await conn.execute(bind_stmt)

        if result.rowcount == 0:
            return None
        
        row = result.fetchone()
        user = UserData(id=row[0], name=row[1], email=row[2])

        result.close()
        return user

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="알수없는 이유로 서비스 오류가 발생하였습니다.")

# 실제 DB에 회원 정보(이름, 이메일, 해싱된 패스워드)
async def register_user(conn: Connection, name: str, email:str, hashed_password: str):
    try:
        query = f"""
        INSERT INTO user(name, email, hashed_password)
        values (:name, :email, :hashed_password)
        """
        stmt = text(query)
        bind_stmt = stmt.bindparams(name=name, email=email, hashed_password=hashed_password)
        
        await conn.execute(bind_stmt)
        await conn.commit()

    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")
    
# 해당 이메일 가진 회원정보 가져오기
async def authenticate_user(conn: Connection, email: str):
    """로그인 시 이메일을 검증하여 일치하면 유저 정보를 반환합니다."""
    try:
        query = """
        SELECT id, name, email, hashed_password FROM user
        WHERE email = :email
        """
        stmt = text(query).bindparams(email=email)
        result = await conn.execute(stmt)

        if result.rowcount == 0:
            return False 
        
        row = result.fetchone()
        
        user = UserDataPASS(id=row[0], name=row[1], email=row[2], hashed_password=row[3])

        result.close()
        return user
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="요청데이터가 제대로 전달되지 않았습니다")