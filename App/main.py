import os
import sys
import warnings
import logging
from pathlib import Path
from dotenv import load_dotenv

# ------ 상위 폴더 경로 설정 --------
REQUIRED_PACKAGES = ["deepguard", "inference", "explainability", "preprocess"]

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent 

for pkg in REQUIRED_PACKAGES:
    pkg_path = str(project_root / pkg)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

# ------ .env 파일 가져오기 ------
load_dotenv()

# -------- Huggingface_Hub 인증 ----------
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
else:
    print("[Warning] HF_TOKEN is not set")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
# from starlette.middleware.sessions import SessionMiddleware
from routes import auth, home, inference, image, video, cam
from utils import exc_handler, middleware, common


# 가상 FastAPI 인스턴스 생성
app = FastAPI(lifespan=common.lifespan)

# StaticFile 등록 (이미지, 비디오 파일)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS Setup for cross-origin requests
# FrontEnd: React, BackEnd: FastAPI
origins_str = os.getenv("CORS_ALLOWED_ORIGINS")
allowed_origins = [origin.strip() for origin in origins_str.split(",")]
app.add_middleware(CORSMiddleware,
                   allow_origins=allowed_origins,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   allow_credentials=True,
                   max_age = -1
                   )

# 세션 미들웨어 등록 - Signed Cookie 이용
# SECRET_KEY = os.getenv("SECRET_KEY", "unique_secret_key")
# app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=3600)

# 세션 미들웨어 등록 - Redis 이용
app.add_middleware(middleware.RedisSessionMiddleware, max_age=3600)

# 라우터 등록 (View: Router, Controller: Service)
app.include_router(auth.router) # 로그인, 로그아웃, 회원가입
app.include_router(home.router) # 세션 유저 정보 가져오기
app.include_router(inference.router) # 이미지, 비디오 비동기 추론 접수, 추론 값 가져오기 
app.include_router(image.router) # 이미지 삭제 및 가져오기 
app.include_router(video.router) # 비디오 삭제 및 가져오기
app.include_router(cam.router) # 딥페이크 이미지, 비디오 위조 흔적 표시

# 커스텀 예외 처리: HTTPException
app.add_exception_handler(StarletteHTTPException, exc_handler.custom_http_exception_handler)
# 커스텀 예외 처리: RequestValidationError
app.add_exception_handler(RequestValidationError, exc_handler.validation_exception_handler)