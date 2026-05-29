from fastapi import status, Request
from fastapi.exceptions import HTTPException

def get_session_user_opt(request:Request) -> dict | None:
    """세션에서 로그인 사용자 정보를 조회한다. 비로그인 상태면 None을 반환한다.

    로그인 여부와 관계없이 접근 가능한 엔드포인트에 사용한다.

    Args:
        request (Request): FastAPI 요청 객체.

    Returns:
        dict | None: 로그인 상태면 {"id": int, "name": str, "email": str} 반환,
            비로그인 상태면 None.
    """
    if "session_user" in request.state.session:
        return request.state.session["session_user"]
    

def get_session_user_prt(request:Request) -> dict:
    """세션에서 로그인 사용자 정보를 조회한다. 비로그인 상태면 401을 raise한다.

    로그인이 필수인 엔드포인트의 Depends 의존성으로 사용한다.

    Args:
        request (Request): FastAPI 요청 객체.

    Returns:
        dict: {"id": int, "name": str, "email": str}

    Raises:
        HTTPException 401: 세션에 사용자 정보가 없을 때.
    """
    if "session_user" not in request.state.session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="해당 서비스는 로그인이 필요합니다.")
    
    return request.state.session["session_user"]