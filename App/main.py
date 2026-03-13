from fastapi import FastAPI
from routes import auth
from utils.common import lifespan

# 가상 인스턴스 생성
app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)