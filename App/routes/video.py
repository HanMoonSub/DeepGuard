from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from services import session_svc, video_svc, artifact_svc
from sqlalchemy import Connection
from db.database import context_get_conn 
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/video", tags=["video"])

# Fetch the user's full video upload history
@router.get("/history", status_code=status.HTTP_200_OK, response_class=JSONResponse,summary="비디오 히스토리 전체 조회")
async def get_video_histories(
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # login required
):
        
    user_id = session_user['id']
    video_histories = await video_svc.get_user_histories(conn, user_id)
    
    return {
        "message": "비디오 히스토리 내역을 성공적으로 불러왔습니다", 
        "status": "success",
        "context": video_histories
    }

# Fetch the details of a single video
@router.get("/history/{video_id}", status_code=status.HTTP_200_OK,response_class=JSONResponse, summary="비디오 개별 상세 조회")
async def get_video_history(
    video_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_opt)
):
    
    user_id = session_user['id'] if session_user else None
    # Pass user_id as well so users can only see their own data
    history = await video_svc.get_user_history(conn, user_id, video_id)
    
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 비디오 기록을 찾을 수 없습니다.")
        
    return {
        "message": "비디오 개별 내역을 성공적으로 불러왔습니다",
        "status": "success",
        "context": history
    }

# Delete a video history entry
@router.delete("/history/{video_id}", status_code=status.HTTP_200_OK,
                response_class=JSONResponse, summary="비디오 버튼 히스토리 삭제")
async def delete_video_history(
    video_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # login required
):
    user_id = session_user['id']
    
    history = await video_svc.get_user_history(conn, user_id, video_id)
    
    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 비디오 기록을 찾을 수 없습니다.")
    
    if history.status == "SUCCESS":
        await video_svc.delete_video_meta_result(conn, video_id)
        await video_svc.delete_video_frame_result(conn, video_id)
        await artifact_svc.delete_artifact(conn, "video", video_id)

    await video_svc.delete_video_db(conn, video_id)
    await video_svc.delete_video(history.video_loc)
    
    return {
        "message": "비디오 내역이 성공적으로 삭제되었습니다",
        "status": "success"
    }