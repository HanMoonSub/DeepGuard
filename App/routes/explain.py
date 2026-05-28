import os
from fastapi import APIRouter, status, Depends
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import Connection
from db.database import context_get_conn
from schemas.explain_schema import ExplainImageRequest, ExplainFrameRequest
from services import session_svc, explain_svc, image_svc, video_svc
from celery_app import celery_app
from celery.result import AsyncResult

router = APIRouter(prefix="/explain", tags=["explain"])

@router.post("/image/{image_id}", status_code=status.HTTP_202_ACCEPTED, 
             response_class=JSONResponse, summary="딥페이크 이미지 위조 흔적 시각화 비동기 접수")
async def explain_image(
    image_id: int,
    explain_req: ExplainImageRequest,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt), # 로그인 필수 
):  
    # 딥페이크 이미지 추론 결과 가져오기 
    result = await image_svc.get_image_result(conn, image_id)
      
    if result.status != "SUCCESS":
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "이미지 위조 흔적 분석은 추론이 성공한 이미지만 가능합니다"
        )

    image_path = "." + result.image_loc
    
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail =  f"요청하신 이미지 파일을 찾을 수 없습니다. 삭제하였는지 다시 확인해주세요."
        )
        
    if result.model_type == "pro" and explain_req.aug_smooth:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pro 모델은 aug_smooth 기능을 지원하지 않습니다",
        )
        
    # Celery Task 호출(Redis Broker 활용)
    task = explain_svc.process_explain_image_task.delay(
                                user_email = session_user["email"],
                                version_type = result.version_type,
                                domain_type = result.domain_type,
                                image_loc = result.image_loc,
                                image_id = result.image_id,
                                category = 1 if result.label == "FAKE" else 0,
                                explain_req_dict = explain_req.model_dump())
    return {
        "message": "딥페이크 이미지 위조 흔적 시각화 접수 완료. 시각화 분석 시작 ...",
        "task_id": task.id, 
    }
    
    
@router.get("/image/result/{task_id}", status_code=status.HTTP_200_OK,
            response_class=JSONResponse, summary="딥페이크 이미지 위조 흔적 시각화 결과 가져오기")
async def get_explain_image_result(
        task_id: str,
        session_user = Depends(session_svc.get_session_user_prt), # 로그인 필수
        ):
    
    # Redis Broker에서 Task ID에 해당하는 비동기 작업 상태 가져오기
    task = AsyncResult(task_id, app=celery_app)
    
    if task.state in ("PENDING", "STARTED", "RETRY"):
        return JSONResponse(
            status_code = status.HTTP_202_ACCEPTED,
            content = {"message": "딥페이크 이미지 위조 흔적 시각화 분석 중 ..."}
        )

    if task.state == "FAILURE":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="딥페이크 이미지 위조 흔적 시각화 중 알 수 없는 오류가 발생하였습니다")
    
    # Celery Task 결과 가져오기
    result = task.result
    
    if result["status"] == "FAILED":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=result["message"])

    return {
        "status": result["status"],
        "message": result["message"],
        "cam_loc": result["cam_loc"],   
    }
  
@router.post("/video/{video_id}/frame/{frame_index}", status_code=status.HTTP_202_ACCEPTED,
             response_class=JSONResponse, summary="딥페이크 비디오 프레임 위조 흔적 시각화 비동기 접수")  
async def explain_frame(
    video_id: int,
    frame_index: int, 
    explain_req: ExplainFrameRequest,
    conn: Connection = Depends(context_get_conn),
    session_user = Depends(session_svc.get_session_user_prt), # 로그인 필수
):
    # 딥페이크 비디오 추론 결과 가져오기 
    result = await video_svc.get_video_result(conn, video_id)
    
    # 딥페이크 비디오 추론 성공 여부 확인하기
    if result.status != "SUCCESS":
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "비디오 프레임 위조 흔적 분석은 추론이 성공한 비디오에서만 가능합니다"
        )

    # 비디오 파일 저장 경로 가져오기
    video_path = "." + result.video_loc
    if not os.path.exists(video_path):
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail =  f"요청하신 비디오 파일을 찾을 수 없습니다. 삭제하였는지 다시 확인해주세요."
        )
    
    # 딥페이크 비디오 프레임 위조 흔적 분석 (pro model는 aug_smooth 사용 불가, 연산이 너무 많아짐)
    if result.model_type == "pro" and explain_req.aug_smooth:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pro 모델은 aug_smooth 기능을 지원하지 않습니다",
        )
    
    # 비디오 내 해당 frame이 몇초에 위치한 frame인지 확인
    frame_time = video_svc.get_video_frame_by_index(conn, video_id, frame_index)
        
    # Celery Task 호출(Redis Broker 활용)
    task = explain_svc.process_explain_frame_task.delay(
                                user_email = session_user["email"],
                                version_type = result.version_type,
                                domain_type = result.domain_type,
                                video_loc = result.video_loc,
                                video_id = video_id,
                                category = 1 if result.label == "FAKE" else 0,
                                frame_time = frame_time, 
                                explain_req_dict = explain_req.model_dump())
    return {
        "message": "딥페이크 비디오 프레임 위조 흔적 시각화 접수 완료. 시각화 분석 시작 ...",
        "task_id": task.id, 
    }

@router.get("/frame/result/{task_id}", status_code=status.HTTP_200_OK,
            response_class=JSONResponse, summary="딥페이크 비디오 프레임 위조 흔적 시각화 결과 가져오기")
async def get_explain_frame_result(
        task_id: str,
        session_user = Depends(session_svc.get_session_user_prt), # 로그인 필수
        ):
    
    # Redis Broker에서 Task ID에 해당하는 비동기 작업 상태 가져오기
    task = AsyncResult(task_id, app=celery_app)
    
    # 비동기 작업 진행 상태 Check
    if task.state in ("PENDING", "STARTED", "RETRY"):
        return JSONResponse(
            status_code = status.HTTP_202_ACCEPTED,
            content = {"message": "딥페이크 비디오 프레임 위조 흔적 시각화 분석 중 ..."}
        )

    # 비동기 작업 실패
    if task.state == "FAILURE":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="딥페이크 비디오 프레임 위조 흔적 시각화 중 알 수 없는 오류가 발생하였습니다")
    
    # Celery Task 결과 가져오기
    result = task.result
    
    # 딥페이크 비디오 프레임 위조 흔적 시각화 생성 또는 파일 저장 도중 오류 발생
    if result["status"] == "FAILED":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=result["message"])

    return {
        "status": result["status"],
        "message": result["message"],
        "cam_loc": result["cam_loc"],   
    }
  