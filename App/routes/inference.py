from fastapi import (
    APIRouter, Depends, status, Form,
    File, UploadFile
)
from fastapi.exceptions import HTTPException
from services import image_svc, session_svc, inference_svc
from sqlalchemy import Connection
from db.database import context_get_conn


router = APIRouter(prefix="/inference", tags=["inference"])

# ---- 딥페이크 비동기 이미지 추론 API ------
@router.post("/image", status_code=status.HTTP_200_OK) # 해당 API 정상 작동시, 200 반환
async def predict_image(imagefile: UploadFile = File(...), # 사용자가 업로드한 이미지 객체
                        version_type: str = Form(...), # deepguard1, deepguard2
                        model_type: str = Form(...), # fast model, pro moel 
                        domain_type: str = Form(...), # model 학습시 사용한 dataset 종류
                        conn: Connection = Depends(context_get_conn), # 이미지 File 저장
                        session_user = Depends(session_svc.get_session_user_opt) # Signed Cookie 없을 시 None 반환
                        ):
        
    user_id = None
    user_email = None
    
    if session_user:    
        user_id = session_user['id']
        user_email = session_user['email']    
    
    # 이미지 업로드 이후, 이미지 저장 경로 반환
    image_loc = await image_svc.upload_image(user_email, imagefile) 
    
    # 이미지 메타데이터 DB 저장
    await image_svc.register_user_image(conn, user_id, image_loc)
        
    # # 이미지 비동기 추론, DeepFake 결과값 반환
    result = await inference_svc.predict_image(image_loc, version_type, model_type, domain_type)
    
    # 로그인 시, 추론 결과값 DB에 저장하기
    if session_user:
        analysis = result["analysis"]; result_msg = result["message"]
        await inference_svc.register_image_result(conn, user_id, image_loc, analysis["prob"], analysis["face_conf"], analysis["face_ratio"],
                                                  analysis["face_brightness"], version_type, model_type, domain_type, result_msg)
    # else: 
        # 비로그인 추론 결과값 반환만 하고 서버 내 이미지 파일 삭제
        # await image_svc.delete_image(conn)
        
    return result
    
        