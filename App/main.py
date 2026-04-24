import os
import sys
from pathlib import Path
import warnings
import logging
from dotenv import load_dotenv

# ------ 상위 폴더 경로 설정 --------
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # App의 상위 폴더

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

# -------- Huggingface_Hub 인증 ----------
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from routes import auth, home, inference, image, video
from utils.common import lifespan
from utils import exc_handler


# 가상 인스턴스 생성
app = FastAPI(lifespan=lifespan)

# StaticFile 등록하기
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cross Origin Resource Sharing
origins_str = os.getenv("CORS_ALLOWED_ORIGINS")
allowed_origins = [origin.strip() for origin in origins_str.split(",")]

app.add_middleware(CORSMiddleware,
                   allow_origins=allowed_origins,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   allow_credentials=True,
                   max_age = -1
                   )

# 세션 미들웨어 등록
SECRET_KEY = os.getenv("SECRET_KEY")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=3600)

# 라우터 등록
app.include_router(auth.router)
app.include_router(home.router)
app.include_router(inference.router)
app.include_router(image.router)
app.include_router(video.router)

# Custom HTTPException Handler
app.add_exception_handler(StarletteHTTPException, exc_handler.custom_http_exception_handler)
# Custom RequestValidationError Handler
app.add_exception_handler(RequestValidationError, exc_handler.validation_exception_handler)