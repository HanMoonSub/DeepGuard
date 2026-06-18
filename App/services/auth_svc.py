from fastapi import status
from fastapi.exceptions import HTTPException
from passlib.context import CryptContext
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from schemas.auth_schema import UserData, UserDataPASS

# 비밀번호 암호화 및 검증
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_hashed_password(password: str) -> str:
    """평문 비밀번호를 bcrypt 알고리즘으로 해싱한다.

    Args:
        password (str): 사용자가 입력한 평문 비밀번호.

    Returns:
        str: bcrypt 해싱된 비밀번호 문자열.
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """평문 비밀번호와 해시된 비밀번호의 일치 여부를 검증한다.

    Args:
        plain_password (str): 사용자가 입력한 평문 비밀번호.
        hashed_password (str): DB에 저장된 bcrypt 해시 비밀번호.

    Returns:
        bool: 비밀번호 일치 여부.
    """
    return pwd_context.verify(plain_password, hashed_password)

async def get_user_by_email(conn: Connection, email: str) -> UserData | None:
    """이메일로 사용자를 조회해 중복 가입 여부를 확인한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        email (str): 조회할 이메일 주소.

    Returns:
        UserData | None: 이메일이 존재하면 UserData 반환, 없으면 None.

    Raises:
        HTTPException 503: DB 쿼리 중 SQLAlchemy 오류 발생 시.
        HTTPException 500: 예기치 못한 예외 발생 시.
    """
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

async def register_user(conn: Connection, name: str, email:str, hashed_password: str) -> None:
    """신규 사용자 정보를 DB에 저장한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        name (str): 사용자 이름.
        email (str): 사용자 이메일 주소.
        hashed_password (str): bcrypt 해시 처리된 비밀번호.

    Raises:
        HTTPException 400: DB INSERT 실패 또는 잘못된 요청 데이터.
    """
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
    
async def authenticate_user(conn: Connection, email: str) -> UserDataPASS | bool:
    """로그인 시 이메일로 사용자를 조회해 인증 정보를 반환한다.

    Args:
        conn (Connection): SQLAlchemy 비동기 DB 커넥션.
        email (str): 로그인 요청 이메일.

    Returns:
        UserDataPASS | bool: 사용자가 존재하면 UserDataPASS 반환, 없으면 False.

    Raises:
        HTTPException 400: DB 쿼리 중 SQLAlchemy 오류 발생 시.
    """
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