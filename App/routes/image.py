from fastapi import APIRouter, Depends, status, HTTPException
from services import session_svc, image_svc
from sqlalchemy import Connection
from db.database import context_get_conn 

router = APIRouter(prefix="/image", tags=["image"])

# 사용자 전체 업로드 히스토리 조회 
@router.get("/history")
async def get_user_histories(
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_opt)
):
    if not session_user:
        return {"message": "해당 history창은 로그인 시 가능합니다", "status": "fail"}
    
    user_id = session_user['id']
    user_histories = await image_svc.get_user_histories(conn, user_id)
    
    return {
        "message": "사용자 내역을 성공적으로 불러왔습니다", 
        "status": "success",
        "context": user_histories
    }

# 특정 이미지의 상세 내역 조회 (개별 히스토리)
@router.get("/history/{image_id}")
async def get_user_individual_history(
    image_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_opt)
):
    if not session_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.")
    
    user_id = session_user['id']
    history = await image_svc.get_user_individual_history(conn, user_id, image_id)
    
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 기록을 찾을 수 없습니다.")
        
    return {
        "message": "개별 내역을 성공적으로 불러왔습니다",
        "status": "success",
        "context": history
    }