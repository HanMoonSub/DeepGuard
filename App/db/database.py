import os
from dotenv import load_dotenv
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager


load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_CONN = f"mysql+aiomysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine: AsyncEngine = create_async_engine(
    url = DATABASE_CONN,
    # echo = True,
    pool_size = 10,
    max_overflow = 2, 
    pool_recycle = 3600 # 1hour 
)

# Celery worker 전용 engine 추가
celery_engine: AsyncEngine = create_async_engine(
    url=DATABASE_CONN,
    poolclass=NullPool  # 풀 없이 매 연결마다 새로 생성
)

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
            
@asynccontextmanager
async def background_db_conn():
    conn = await celery_engine.connect()  # celery_engine으로 교체
    try:
        yield conn
    finally:
        await conn.close()
    