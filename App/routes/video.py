from fastapi import APIRouter, Depends, status, HTTPException
from services import session_svc, video_svc
from sqlalchemy import Connection
from db.database import context_get_conn 

router = APIRouter(prefix="/video", tags=["video"])

# 사용자 전체 비디오 업로드 히스토리 조회 
@router.get("/history", status_code=status.HTTP_200_OK)
async def get_video_histories(
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_opt)
):
    if not session_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.")
        
    user_id = session_user['id']
    video_histories = await video_svc.get_user_histories(conn, user_id)
    
    return {
        "message": "비디오 히스토리 내역을 성공적으로 불러왔습니다", 
        "status": "success"
    }

# 특정 비디오의 상세 내역 조회
@router.get("/history/{video_id}", status_code=status.HTTP_200_OK)
async def get_video_history(
    video_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_opt)
):
    if not session_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다.")
    
    user_id = session_user['id']
    # 본인의 데이터만 조회할 수 있도록 user_id도 함께 전달
    history = await video_svc.get_user_history(conn, user_id, video_id)
    
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 비디오 기록을 찾을 수 없습니다.")
        
    return {
        "message": "비디오 개별 내역을 성공적으로 불러왔습니다",
        "status": "success"
    }