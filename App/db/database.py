import os
from dotenv import load_dotenv
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()
DATABASE_CONN = os.getenv("DATABASE_CONN")
print("database_conn: ", DATABASE_CONN)

engine: AsyncEngine = create_async_engine(
    url = DATABASE_CONN,
    # echo = True,
    pool_size = 10,
    max_overflow = 2, 
    pool_recycle = 3600 # 1hour 
)

async def direct_get_conn():
    conn = None
    try:
        conn = await engine.connect()
        return conn
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다")

async def context_get_conn():
    conn = None
    try:
        conn = await engine.connect()
        yield conn
    except SQLAlchemyError as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="요청하신 서비스가 잠시 내부적으로 문제가 발생하였습니다")
    finally:
        if conn:
            await conn.close()
    