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

# 데이터베이스 통신 (조회, 가입, 인증)
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
                        
async def register_user(conn: Connection, name: str, email:str, hashed_password: str):
    try:
        query = f"""
        INSERT INTO user(name, email, hashed_password)
        values ('{name}', '{email}', '{hashed_password}')
        """
        print("query:", query)
        await conn.execute(text(query))
        await conn.commit()

    except SQLAlchemyError as e:
        print(e)
        await conn.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다.")
    
# 비즈니스 로직 (로그인 시 인증 맞는지 확인)
async def authenticate_user(conn: Connection, email: str, password: str):
    """로그인 시 이메일과 비밀번호를 검증하여 일치하면 유저 정보를 반환합니다."""
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
        db_hashed_password = row[3]
        
        
        if not verify_password(password, db_hashed_password):
            return False 
            
        
        return UserData(id=row[0], name=row[1], email=row[2])

    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="데이터베이스 조회 중 오류가 발생했습니다.")