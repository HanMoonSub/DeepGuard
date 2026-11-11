from fastapi import status, Request
from fastapi.exceptions import HTTPException

def get_session_user_opt(request:Request) -> dict | None:
    """Get the logged-in user from the session. Returns None when not logged in.

    Used for endpoints that are accessible whether or not the user is logged in.

    Args:
        request (Request): FastAPI request object.

    Returns:
        dict | None: {"id": int, "name": str, "email": str} when logged in,
            None when not logged in.
    """
    if "session_user" in request.state.session:
        return request.state.session["session_user"]
    

def get_session_user_prt(request:Request) -> dict:
    """Get the logged-in user from the session. Raises 401 when not logged in.

    Used as a Depends dependency for endpoints that require login.

    Args:
        request (Request): FastAPI request object.

    Returns:
        dict: {"id": int, "name": str, "email": str}

    Raises:
        HTTPException 401: When the session has no user info.
    """
    if "session_user" not in request.state.session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="해당 서비스는 로그인이 필요합니다.")
    
    return request.state.session["session_user"]