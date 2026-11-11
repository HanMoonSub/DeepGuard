from fastapi import status
from fastapi.exceptions import HTTPException
from passlib.context import CryptContext
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from schemas.auth_schema import UserData, UserDataPASS

# Password hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_hashed_password(password: str) -> str:
    """Hash a plain-text password with bcrypt.

    Args:
        password (str): Plain-text password entered by the user.

    Returns:
        str: bcrypt-hashed password string.
    """
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check whether a plain-text password matches a hashed password.

    Args:
        plain_password (str): Plain-text password entered by the user.
        hashed_password (str): bcrypt hash stored in the DB.

    Returns:
        bool: Whether the passwords match.
    """
    return pwd_context.verify(plain_password, hashed_password)

async def get_user_by_email(conn: Connection, email: str) -> UserData | None:
    """Look up a user by email to check for a duplicate sign-up.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        email (str): Email address to look up.

    Returns:
        UserData | None: UserData if the email exists, otherwise None.

    Raises:
        HTTPException 503: When a SQLAlchemy error occurs during the DB query.
        HTTPException 500: When an unexpected exception occurs.
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
    """Save a new user to the DB.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        name (str): User name.
        email (str): User email address.
        hashed_password (str): bcrypt-hashed password.

    Raises:
        HTTPException 400: When the DB INSERT fails or the request data is invalid.
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
    """Look up a user by email at login and return their authentication info.

    Args:
        conn (Connection): SQLAlchemy async DB connection.
        email (str): Email from the login request.

    Returns:
        UserDataPASS | bool: UserDataPASS if the user exists, otherwise False.

    Raises:
        HTTPException 400: When a SQLAlchemy error occurs during the DB query.
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