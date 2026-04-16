from fastapi import (
    APIRouter, Depends, status, Form,
    File, UploadFile, BackgroundTasks
)
from fastapi.exceptions import HTTPException
from services import image_svc, session_svc, inference_svc, video_svc
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
        
    # # 이미지 비동기 추론, DeepFake 결과값 반환
    result = await inference_svc.predict_image(image_loc, version_type, model_type, domain_type)
    
    # 로그인 시, 추론 결과값 DB에 저장하기
    if session_user:
        analysis = result["analysis"]; result_msg = result["message"]
        await inference_svc.register_image_result(conn, user_id, image_loc, analysis["prob"], analysis["face_conf"], analysis["face_ratio"],
                                                  analysis["face_brightness"], version_type, model_type, domain_type, result_msg)
    else: 
        # 비로그인 추론 결과값 반환만 하고 서버 내 이미지 파일 삭제
        await image_svc.delete_image(image_loc)
        
    return result

# ---- 딥페이크 비동기 비디오 추론 API ------
@router.post("/video", status_code=status.HTTP_202_ACCEPTED) # 해당 API 요청 시, 접수 완료(분석은 백그라운드에서 실행)
async def predict_video(
                        background_tasks: BackgroundTasks, # 백그라운드에서 실행
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
    video_id = await inference_svc.register_video_result(conn, user_id, video_loc, version_type, model_type, domain_type)
    print("video_id: ", video_id)
    
    background_tasks.add_task(
        inference_svc.process_video_task,
        video_id, video_loc, version_type, model_type, domain_type, user_id
    )
            
    return {
        "video_id": video_id, 
        "message": "비디오 업로드 성공. 비디오 분석 시작 ...",
        "status": "success",
    }

# ---- 딥페이크 추론 결과값 불러오기 API ----- 
@router.get("/video/result/{video_id}", status_code=status.HTTP_200_OK)
async def get_video_result(
                           video_id: int,
                           conn: Connection = Depends(context_get_conn),
                           session_user = Depends(session_svc.get_session_user_opt)
                           ):
    
    video_data = await inference_svc.get_video_result(conn, video_id)  
    print("video data: ", video_data)
    
    if video_data.status == 'FAILED':
        if session_user:
            await video_svc.delete_video(video_data.video_loc) # 서버 내 저장 파일 삭제
            # await video_svc.detle_video_db # db에서도 삭제 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=video_data.result_msg  
        )
    
    return video_data