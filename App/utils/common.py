from fastapi import FastAPI
from db.database import engine
from contextlib import asynccontextmanager

# Uvicorn이 앱 시작/종료 시 asynccontextmanager를 호출함
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # ── Startup ──────────────────────────────────────────
    # Event Loop 위에서 실행, DB 연결 풀 초기화
    print("Starting up...")
    
    # yield 이후 app이 실제 요청을 처리하기 시작함
    yield
    
    # ── Shutdown ─────────────────────────────────────────
    # 새 요청은 차단되고, 진행 중인 요청 완료 후 여기로 진입
    print("Shutting down...")
    
    # SQLAlchemy 비동기 엔진의 커넥션 풀 전체 반환 및 종료
    await engine.dispose()