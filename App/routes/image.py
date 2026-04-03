from fastapi import APIRouter, Depends, status
from services import auth_svc, session_svc, image_svc
from sqlalchemy import Connection
from db.database import context_get_conn

router = APIRouter(prefix="/image", tags=["image"])

@router.get("/history")
async def get_user_histories(conn: Connection = Depends(context_get_conn),
                             session_user = Depends(session_svc.get_session_user_opt)):
    
    if not session_user:
        return {"message": "해당 history창은 로그인 시 가능합니다", "status": "fail"}
    
    user_id = session_user['id']
    user_histories = await image_svc.get_user_histories(conn, user_id)
    for history in user_histories:
        print("image_loc: {}, create_at: {}".format(history.image_loc, history.created_at))
    
    return {"message": "사용자 내역을 성공적으로 불러왔습니다", 
            "status": "success",
            "context": user_histories}