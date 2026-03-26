from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# HTTPException 오류 발생 시 커스텀 exception handler 로직
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        content = {
            "error_type" : "http",
            "status_code": exc.status_code,
            "title_message": "불편을 드려 죄송합니다",
            "detail": exc.detail},
        status_code = exc.status_code
    )
    
# RequestValidationError 오류 발생 시 커스텀 exception handler 로직
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        content = {
            "error_type" : "vaild",
            "status_code": status.HTTP_422_UNPROCESSABLE_CONTENT,
            "title_message": "잘못된 값을 입력하였습니다",
            "detail": exc.errors()
            },
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    )