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
    # 에러 메시지 가공: "필드명: 에러내용" 형태로 요약
    # 예: "email: value is not a valid email address"
    error_details = [f"{err['loc'][-1]}: {err['msg']}" for err in exc.errors()]
    summary_detail = " | ".join(error_details) # 여러 에러를 한 줄로 합침

    return JSONResponse(
        content = {
            "error_type" : "valid", # 오타 수정: vaild -> valid
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "title_message": "입력값이 올바르지 않습니다",
            "detail": summary_detail # 요약된 문자열 전달
            },
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    )