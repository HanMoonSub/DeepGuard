import io
from fastapi import APIRouter, status, Depends
from fastapi.exceptions import HTTPException
from utils.responses import PNGStreamingResponse
from sqlalchemy import Connection
from db.database import context_get_conn
from schemas.explain_schema import ExplainRequest
from services import session_svc, explain_svc, image_svc

router = APIRouter(prefix="/explain", tags=["explain"])

@router.post("/image/{image_id}", status_code=status.HTTP_200_OK, 
             response_class=PNGStreamingResponse , summary="딥페이크 이미지 위조 흔적 시각화(CAM)")
async def explain_image(
    image_id: int,
    explain_req: ExplainRequest,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt), # 로그인 필수 
):
    result = await image_svc.get_image_result(conn, image_id)
    
    if result.status != "SUCCESS":
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "이미지 위조 흔적 분석은 추론이 성공한 이미지만 가능합니다"
        )
    
    img_bytes = await explain_svc.explain_image(conn, result, explain_req)
    return PNGStreamingResponse(io.BytesIO(img_bytes))
    
    