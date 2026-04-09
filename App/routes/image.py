from fastapi import APIRouter, Depends, status, HTTPException
from services import session_svc, image_svc
from sqlalchemy import Connection
from db.database import context_get_conn 

router = APIRouter(prefix="/image", tags=["image"])

# 사용자 전체 업로드 히스토리 조회 
@router.get("/history", status_code=status.HTTP_200_OK)
async def get_user_histories(
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_opt)
):
    
    user_id = session_user['id']
    user_histories = await image_svc.get_user_histories(conn, user_id)
    
    return {
        "message": "사용자 내역을 성공적으로 불러왔습니다", 
        "status": "success",
        "context": user_histories
    }

# 특정 이미지의 상세 내역 조회 (개별 히스토리)
@router.get("/history/{image_id}", status_code=status.HTTP_200_OK)
async def get_user_history(
    image_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_opt)
):
    
    user_id = session_user['id']
    history = await image_svc.get_user_history(conn, user_id, image_id)
    
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 기록을 찾을 수 없습니다.")
        
    return {
        "message": "개별 내역을 성공적으로 불러왔습니다",
        "status": "success",
        "context": history
    }