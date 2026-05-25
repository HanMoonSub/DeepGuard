from pydantic import BaseModel, EmailStr, Field

# [응답용] 프론트엔드 반환용 유저 기본 정보 (보안상 비밀번호 제외)
class UserData(BaseModel):
    id: int = Field(..., description="유저 고유 ID")
    name: str = Field(..., description="유저 이름")
    email: str = Field(..., description="유저 이메일")

# [내부용] DB 조회 및 서버 내부 인증용 (UserData 상속 + 암호화된 비밀번호)
class UserDataPASS(UserData):
    hashed_password: str = Field(..., description="bcrypt로 해싱된 비밀번호 (서버 내부 인증용, 외부 노출 금지)")


# [요청용] 로그인 API (POST /auth/login) 수신 JSON 데이터 검증
class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="로그인 이메일")
    password: str = Field(min_length=8, max_length=30, description="비밀번호 (8자 이상 30자 이하)")

# [요청용] 회원가입 API (POST /auth/register) 수신 JSON 데이터 검증
class RegisterRequest(LoginRequest):
    name: str = Field(min_length=2, max_length=100, description="유저 이름 (2자 이상 100자 이하)")