from fastapi import (
    APIRouter, Depends, status, Form,
    File, UploadFile
)
from fastapi.exceptions import HTTPException
from services import image_svc, session_svc, inference_svc, video_svc
from sqlalchemy import Connection
from db.database import context_get_conn


router = APIRouter(prefix="/inference", tags=["inference"])

# ---- 딥페이크 비동기 이미지 추론 API ------
@router.post("/image", status_code=status.HTTP_202_ACCEPTED) # 해당 API 요청 시, 접수 완료(분석은 백그라운드에서 실행)
async def predict_image(
                        imagefile: UploadFile = File(...), # 사용자가 업로드한 이미지 객체
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
        
    # 빈 이미지 DB 생성 후, image_id 받기
    image_id = await image_svc.register_image_result(conn, user_id, image_loc, version_type, model_type, domain_type)
    
    # Celery Task 호출(Redis 브로커로 작업 전달)
    inference_svc.process_image_task.delay(image_id, image_loc, version_type, model_type, domain_type, user_id)
    
    return {
        "image_id": image_id, 
        "message": "이미지 업로드 성공. 이미지 분석 시작 ...",
        "status": "success",
    }

# ---- 딥페이크 이미지 추론 결과값 불러오기 API ----
@router.get("/image/{image_id}", status_code=status.HTTP_200_OK)
async def get_image_result(
                            image_id: int,
                            conn: Connection = Depends(context_get_conn),
                            session_user = Depends(session_svc.get_session_user_opt)
                            ):
    image_data = await inference_svc.get_image_result(conn, image_id)
    
    if image_data.status == "FAILED":
        if session_user:
            await image_svc.delete_image(image_data.image_loc)
            # await image_svc.delet_image_db # db에서도 삭제
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=image_data.result_msg
        )
    return image_data

# ---- 딥페이크 비동기 비디오 추론 API ------
@router.post("/video", status_code=status.HTTP_202_ACCEPTED) # 해당 API 요청 시, 접수 완료(분석은 백그라운드에서 실행)
async def predict_video(
                        videofile: UploadFile = File(...), # 사용자가 업로드한 비디오 객체
                        version_type: str = Form(...), # deepguard1, deepguard2
                        model_type: str = Form(...), # fast model, pro moel 
                        domain_type: str = Form(...), # model 학습시 사용한 dataset 종류
                        conn: Connection = Depends(context_get_conn), # 비디오 File 저장
                        session_user = Depends(session_svc.get_session_user_opt) # Signed Cookie 없을 시 None 반환
                        ):
    
    user_id = None
    user_email = None
    
    if session_user:    
        user_id = session_user['id']
        user_email = session_user['email'] 
        
    # 비디오 업로드 이후, 비디오 저장 경로 반환
    video_loc = await video_svc.upload_video(user_email, videofile) 
    
    # 빈 비디오 DB 생성 후, video_id 받기
    video_id = await video_svc.register_video_result(conn, user_id, video_loc, version_type, model_type, domain_type)
    
    # Celery Task 호출(Redis 브로커로 작업 전달)
    inference_svc.process_video_task.delay(video_id, video_loc, version_type, model_type, domain_type, user_id)
            
    return {
        "video_id": video_id, 
        "message": "비디오 업로드 성공. 비디오 분석 시작 ...",
        "status": "success",
    }

# ---- 딥페이크 비디오 추론 결과값 불러오기 API ----- 
@router.get("/video/{video_id}", status_code=status.HTTP_200_OK)
async def get_video_result(
                           video_id: int,
                           conn: Connection = Depends(context_get_conn),
                           session_user = Depends(session_svc.get_session_user_opt)
                           ):
    
    video_data = await inference_svc.get_video_result(conn, video_id)  
    
    if video_data.status == 'FAILED':
        if session_user:
            await video_svc.delete_video(video_data.video_loc) # 서버 내 저장 파일 삭제
            # await video_svc.detle_video_db # db에서도 삭제 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=video_data.result_msg  
        )
    
    return video_data

# ---- 딥페이크 상세 분석 (타임라인) 불러오기 API ----
@router.get("/video/{video_id}/detail", status_code=status.HTTP_200_OK)
async def get_video_detail(
                           video_id: int,
                           conn: Connection = Depends(context_get_conn),
                           session_user = Depends(session_svc.get_session_user_opt)
                           ):
    video_detail = await inference_svc.get_video_frame_results(conn, video_id)
    return video_detail