from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from routes import auth
from utils.common import lifespan
from utils import exc_handler

# 가상 인스턴스 생성
app = FastAPI(lifespan=lifespan)

# CORSMiddleware
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"],
                   max_age = -1
                   )

app.include_router(auth.router)

# Custom HTTPException Handler
app.add_exception_handler(StarletteHTTPException, exc_handler.custom_http_exception_handler)
# Custom RequestValidationError Handler
app.add_exception_handler(RequestValidationError, exc_handler.validation_exception_handler)