import os
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from routes import auth, home
from utils.common import lifespan
from utils import exc_handler
from dotenv import load_dotenv

# 가상 인스턴스 생성
app = FastAPI(lifespan=lifespan)

# Cross Origin Resource Sharing
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
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


# Custom HTTPException Handler
app.add_exception_handler(StarletteHTTPException, exc_handler.custom_http_exception_handler)
# Custom RequestValidationError Handler
app.add_exception_handler(RequestValidationError, exc_handler.validation_exception_handler)