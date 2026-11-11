from fastapi import APIRouter, Depends, status, HTTPException
from services import session_svc, image_svc, artifact_svc
from sqlalchemy import Connection
from db.database import context_get_conn 
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/image", tags=["image"])

# Fetch the user's full upload history
@router.get("/history", status_code=status.HTTP_200_OK, response_class=JSONResponse,summary="이미지 히스토리 전체 조회")
async def get_user_histories(
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # login required
):
    
    user_id = session_user['id']
    user_histories = await image_svc.get_user_histories(conn, user_id)
    
    return {
        "message": "사용자 내역을 성공적으로 불러왔습니다", 
        "status": "success",
        "context": user_histories
    }

# Fetch the details of a single image (individual history entry)
@router.get("/history/{image_id}", status_code=status.HTTP_200_OK, response_class=JSONResponse,summary="이미지 개별 상세 조회")
async def get_user_history(
    image_id: int,
    conn: Connection = Depends(context_get_conn),
):
    
    history = await image_svc.get_user_history(conn, image_id)

    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 이미지 기록을 찾을 수 없습니다.")
        
    return {
        "message": "개별 내역을 성공적으로 불러왔습니다",
        "status": "success",
        "context": history
    }

# Delete an image history entry
@router.delete("/history/{image_id}", status_code=status.HTTP_200_OK,response_class=JSONResponse, summary="이미지 버튼 히스토리 삭제")
async def delete_image_history(
    image_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # login required
):
    history = await image_svc.get_user_history(conn, image_id)

    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 이미지 기록을 찾을 수 없습니다.")

    # Delete the DB record (including the per-region artifact result)
    await artifact_svc.delete_artifact(conn, "image", image_id)
    await image_svc.delete_image_db(conn, image_id)
    # Delete the actual image file
    await image_svc.delete_image(history.image_loc)
    
    return {
        "message": "이미지 내역이 성공적으로 삭제되었습니다",
        "status": "success"
    }