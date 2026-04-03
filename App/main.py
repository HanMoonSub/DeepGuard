import os

import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # App의 상위 폴더

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from routes import auth, home, inference, image
from utils.common import lifespan
from utils import exc_handler
from dotenv import load_dotenv


# 가상 인스턴스 생성
app = FastAPI(lifespan=lifespan)

# StaticFile 등록하기
app.mount("/static", StaticFiles(directory="static"), name="static")

# Cross Origin Resource Sharing
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:3000"],
                   allow_methods=["*"],
                   allow_headers=["*"],
                   allow_credentials=True,
                   max_age = -1
                   )

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=3600)

app.include_router(auth.router)
app.include_router(home.router)
app.include_router(inference.router)
app.include_router(image.router)

# Custom HTTPException Handler
app.add_exception_handler(StarletteHTTPException, exc_handler.custom_http_exception_handler)
# Custom RequestValidationError Handler
app.add_exception_handler(RequestValidationError, exc_handler.validation_exception_handler)