from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


# Custom exception handler logic for HTTPException errors
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        content = {
            "error_type" : "http",
            "status_code": exc.status_code,
            "title_message": "불편을 드려 죄송합니다",
            "detail": exc.detail},
        status_code = exc.status_code
    )
    
# Custom exception handler logic for RequestValidationError errors
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Format the error messages: summarize as "field: error"
    # e.g. "email: value is not a valid email address"
    error_details = [f"{err['loc'][-1]}: {err['msg']}" for err in exc.errors()]
    summary_detail = " | ".join(error_details) # join multiple errors into one line

    return JSONResponse(
        content = {
            "error_type" : "valid", # typo fix: vaild -> valid
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "title_message": "입력값이 올바르지 않습니다",
            "detail": summary_detail # pass the summarized string
            },
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    )