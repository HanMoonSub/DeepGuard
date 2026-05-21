from fastapi import (
    APIRouter, UploadFile, status,
    Depends, Form, File)
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from services import image_svc, session_svc, inference_svc, video_svc
from sqlalchemy import Connection
from db.database import context_get_conn
from schemas.video_schema import VideoDetailResponse, VideoDetailData
from schemas.image_schema import ImageData_indi
from typing import Literal

router = APIRouter(prefix="/inference", tags=["inference"])

@router.post("/image", status_code=status.HTTP_202_ACCEPTED,
             response_class=JSONResponse, summary="딥페이크 비동기 이미지 추론 접수") 
async def predict_image(
                        imagefile: UploadFile = File(...), # 사용자가 업로드한 이미지 객체
                        version_type: Literal["v1","v2"] = Form("v2", description="모델 엔진 버전"),
                        model_type: Literal["fast","pro"] = Form("fast", description="추론 모드 (fast: 속도 우선, pro: 정확도 우선)"), 
                        domain_type: Literal["서양인","동양인"] = Form("서양인", description="학습 데이터셋 도메인"),
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

@router.get("/image/{image_id}", status_code=status.HTTP_200_OK,
            response_model=ImageData_indi, summary="딥페이크 비동기 이미지 추론 결과값 가져오기")
async def get_image_result(
                            image_id: int,
                            conn: Connection = Depends(context_get_conn),
                            session_user = Depends(session_svc.get_session_user_opt)
                            ):
    image_data = await image_svc.get_image_result(conn, image_id)
    
    if image_data.status == "FAILED":
        if session_user:
            await image_svc.delete_image(image_data.image_loc)
            await image_svc.delete_image_db(image_data.image_id)

        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=image_data.result_msg
        )
    return image_data

@router.post("/video", status_code=status.HTTP_202_ACCEPTED, 
             response_class=JSONResponse, summary="딥페이크 비동기 비디오 추론 접수")
async def predict_video(
                        videofile: UploadFile = File(...), # 사용자가 업로드한 비디오 객체
                        version_type: Literal["v1","v2"] = Form("v2", description="모델 엔진 버전"),
                        model_type: Literal["fast","pro"] = Form("fast", description="추론 모드 (fast: 속도 우선, pro: 정확도 우선)"), 
                        domain_type: Literal["서양인","동양인"] = Form("서양인", description="학습 데이터셋 도메인"),
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

@router.get("/video/{video_id}", status_code=status.HTTP_200_OK, 
            response_model=VideoDetailData, summary="딥페이크 비디오 비디오 추론 결과값 가져오기")
async def get_video_result(
                           video_id: int,
                           conn: Connection = Depends(context_get_conn),
                           session_user = Depends(session_svc.get_session_user_opt)
                           ):
    
    video_data = await video_svc.get_video_result(conn, video_id)  
    
    if video_data.status == 'FAILED':

        if session_user:
            await video_svc.delete_video(video_data.video_loc)
            await video_svc.delete_video_db(video_data.video_loc)


        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=video_data.result_msg  
        )
    
    return video_data

@router.get("/video/{video_id}/detail", status_code=status.HTTP_200_OK, 
            response_model=VideoDetailResponse ,summary="딥페이크 상세 분석 결과값 가져오기")
async def get_video_detail(
                           video_id: int,
                           conn: Connection = Depends(context_get_conn),
                           session_user = Depends(session_svc.get_session_user_prt) # 로그인 필수
                           ):
    video_data = await video_svc.get_video_result(conn, video_id)
    if video_data.status != "SUCCESS":
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = f"비디오 상세 분석은 추론이 성공한 비디오만 가능합니다"
        )
    
    meta = await video_svc.get_video_meta_result(conn, video_id)
    frames = await video_svc.get_video_frame_result(conn, video_id)

    return VideoDetailResponse(meta=meta, frames=frames)
