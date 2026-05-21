from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from services import session_svc, video_svc
from sqlalchemy import Connection
from db.database import context_get_conn 
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/video", tags=["video"])

# 사용자 전체 비디오 업로드 히스토리 조회 
@router.get("/history", status_code=status.HTTP_200_OK, response_class=JSONResponse,summary="비디오 히스토리 전체 조회")
async def get_video_histories(
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # 로그인 필수
):
        
    user_id = session_user['id']
    video_histories = await video_svc.get_user_histories(conn, user_id)
    
    return {
        "message": "비디오 히스토리 내역을 성공적으로 불러왔습니다", 
        "status": "success",
        "context": video_histories
    }

# 특정 비디오의 상세 내역 조회
@router.get("/history/{video_id}", status_code=status.HTTP_200_OK,response_class=JSONResponse, summary="비디오 개별 상세 조회")
async def get_video_history(
    video_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # 로그인 필수
):
    
    user_id = session_user['id']
    # 본인의 데이터만 조회할 수 있도록 user_id도 함께 전달
    history = await video_svc.get_user_history(conn, user_id, video_id)
    
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 비디오 기록을 찾을 수 없습니다.")
        
    return {
        "message": "비디오 개별 내역을 성공적으로 불러왔습니다",
        "status": "success",
        "context": history
    }

# 비디오 히스토리 삭제
@router.delete("/history/{video_id}", status_code=status.HTTP_200_OK,
                response_class=JSONResponse, summary="비디오 버튼 히스토리 삭제")
async def delete_video_history(
    video_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # 로그인 필수
):
    user_id = session_user['id']
    
    history = await video_svc.get_user_history(conn, user_id, video_id)
    
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 비디오 기록을 찾을 수 없습니다.")
    
    if history.status == "SUCCESS":
        await video_svc.delete_video_meta_result(conn, video_id)
        await video_svc.delete_video_frame_result(conn, video_id)

    await video_svc.delete_video_db(conn, video_id)
    await video_svc.delete_video(history.video_loc)
    
    return {
        "message": "비디오 내역이 성공적으로 삭제되었습니다",
        "status": "success"
    }