from fastapi import FastAPI
from routes import auth

# 가상 인스턴스 생성
app = FastAPI()

app.include_router(auth.router)