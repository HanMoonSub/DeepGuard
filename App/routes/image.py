from fastapi import APIRouter, Depends, status, HTTPException
from services import session_svc, image_svc
from sqlalchemy import Connection
from db.database import context_get_conn 

router = APIRouter(prefix="/image", tags=["image"])

# 사용자 전체 업로드 히스토리 조회 
@router.get("/history", status_code=status.HTTP_200_OK)
async def get_user_histories(
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # 로그인 필수
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
):
    
    history = await image_svc.get_user_history(conn, image_id)

    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 이미지 기록을 찾을 수 없습니다.")
        
    return {
        "message": "개별 내역을 성공적으로 불러왔습니다",
        "status": "success",
        "context": history
    }

# 이미지 히스토리 삭제
@router.delete("/history/{image_id}", status_code=status.HTTP_200_OK, summary="버튼 삭제")
async def delete_image_history(
    image_id: int,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt) # 로그인 필수
):
    history = await image_svc.get_user_history(conn, image_id)

    if not history:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 이미지 기록을 찾을 수 없습니다.")

    # DB 레코드 삭제 
    await image_svc.delete_image_db(conn, image_id)
    # 실제 이미지 삭제
    await image_svc.delete_image(history.image_loc)
    
    return {
        "message": "이미지 내역이 성공적으로 삭제되었습니다",
        "status": "success"
    }