from pydantic import BaseModel, EmailStr, Field

class UserData(BaseModel):
    """
    [Response] Basic user info returned to the frontend (password excluded for security)
    """
    id: int = Field(..., description="유저 고유 ID")
    name: str = Field(..., description="유저 이름")
    email: str = Field(..., description="유저 이메일")

class UserDataPASS(UserData):
    """
    [Internal] For DB lookups and server-side authentication (extends UserData + hashed password)
    """
    hashed_password: str = Field(..., description="bcrypt로 해싱된 비밀번호 (서버 내부 인증용, 외부 노출 금지)")

class LoginRequest(BaseModel):
    """
    [Request] Validates the JSON body of the login API (POST /auth/login)
    """
    email: EmailStr = Field(..., description="로그인 이메일")
    password: str = Field(min_length=8, max_length=30, description="비밀번호 (8자 이상 30자 이하)")

class RegisterRequest(LoginRequest):
    """
    [Request] Validates the JSON body of the sign-up API (POST /auth/register)
    """
    name: str = Field(min_length=2, max_length=100, description="유저 이름 (2자 이상 100자 이하)")