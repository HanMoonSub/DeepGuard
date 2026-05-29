import cv2
import asyncio
import numpy as np
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import Connection
from services import image_svc, inference_svc
from explainability import (
    CAMExplainer, 
    HiResCAMExplainer, GradCAMElementWiseExplainer, LayerCAMExplainer,
    EigenGradCAMExplainer, GradCAMPlusPlusExplainer, XGradCAMExplainer,
)
from celery_app import celery_app

# LOW branch: 국소 위조 흔적 포착 / HIGH branch: 전역 위조 흔적 포착 
EXPLAINER_REGISTRY = {
    "hirescam":   HiResCAMExplainer, # LOW branch
    "gradcamelementwise": GradCAMElementWiseExplainer, # LOW branch
    "layercam":   LayerCAMExplainer, # LOW branch
    "eigengradcam": EigenGradCAMExplainer, # HIGH branch
    "gradcamplusplus":   GradCAMPlusPlusExplainer, # HIGH branch
    "xgradcam":     XGradCAMExplainer, # HIGH branch
}

_explainer_cache: dict = {}

def _extract_face_from_frame(video_path: str, frame_time: float, explainer: CAMExplainer) -> np.ndarray:
    """비디오의 특정 시간대 프레임에서 얼굴 영역을 추출해 크롭된 배열을 반환한다.

    Args:
        video_path (str): 비디오 파일 경로.
        frame_time (float): 프레임을 추출할 시간 (초 단위).
        explainer (CAMExplainer): 얼굴 감지 및 크롭에 사용할 CAMExplainer 인스턴스.

    Returns:
        np.ndarray: 크롭된 얼굴 이미지 배열 (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
    ret, frame = cap.read()
    cap.release()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbox = explainer._get_face_bbox(frame_rgb)
    
    return explainer._crop_face(frame_rgb, bbox[:4])
    
def _get_or_create_explainer(model_name: str, dataset: str, explain_req_dict: dict) -> CAMExplainer:
    """캐시에서 CAMExplainer를 조회하거나, 없으면 새로 생성해 캐싱 후 반환한다.

    캐시 키는 (model_name, dataset, explainer_type, branch_level) 4개 파라미터로 구성된다.
    추론 시점 파라미터(aug_smooth, threshold 등)는 키에서 제외해 메모리 사용을 최소화한다.

    Args:
        model_name (str): 추론에 사용할 모델 이름.
        dataset (str): 모델 학습에 사용된 데이터셋 이름.
        explain_req_dict (dict): 시각화 요청 파라미터 딕셔너리.
            - explainer_type (str): EXPLAINER_REGISTRY 키값.
            - branch_level (str): 모델 브랜치 레벨 ("LOW" | "HIGH").

    Returns:
        CAMExplainer: 캐시된 또는 새로 생성된 CAMExplainer 인스턴스.
    """
    cache_key = (model_name, dataset, explain_req_dict["explainer_type"], explain_req_dict["branch_level"])
    
    if cache_key not in _explainer_cache:
        _explainer_cache[cache_key] = EXPLAINER_REGISTRY[explain_req_dict["explainer_type"]](
            model_name = model_name,
            dataset = dataset,
            branch_level = explain_req_dict["branch_level"],
        )
    return _explainer_cache[cache_key]
        
def _run_visualization(explainer: CAMExplainer, image_path: str, category: int, explain_req_dict: dict) -> np.ndarray:
    """이미지 경로를 받아 display_type에 따라 시각화 이미지를 생성한다.

    Args:
        explainer (CAMExplainer): 시각화에 사용할 CAMExplainer 인스턴스.
        image_path (str): 입력 이미지 파일 경로.
        category (int): 타겟 클래스 인덱스 (위조: 1, 정상: 0).
        explain_req_dict (dict): 시각화 요청 파라미터 딕셔너리.
            - display_type (str): "heatmap" | "bbox" | "heatmap_bbox"
            - overlay_ratio (float): 히트맵 오버레이 시 원본 이미지 가중치.
            - threshold (float | str): CAM 이진화 임계값.
            - aug_smooth (bool): Test-time augmentation 가동 여부.
            - eigen_smooth (bool): PCA 기반 노이즈 제거 가동 여부.

    Returns:
        np.ndarray: 시각화가 적용된 얼굴 이미지 (RGB, np.uint8).
    """
    if explain_req_dict["display_type"] == "heatmap":
        return explainer.display_heatmap_on_image(image_path, image_weight=explain_req_dict["overlay_ratio"], threshold=explain_req_dict["threshold"],
                                                 category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])
    elif explain_req_dict["display_type"] == "bbox":  
        return explainer.display_bbox_on_image(image_path, threshold=explain_req_dict["threshold"],
                                              category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"]) 
    else: # display_type == "heatmap_bbox"
        return explainer.display_heatmap_bbox_on_image(image_path, image_weight=explain_req_dict["overlay_ratio"], threshold=explain_req_dict["threshold"], 
                                                       category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])

def _run_visualization_from_array(explainer: CAMExplainer, face: np.ndarray, category: int, explain_req_dict: dict) -> np.ndarray:
    """크롭된 얼굴 배열을 받아 display_type에 따라 시각화 이미지를 생성한다.

    Args:
        explainer (CAMExplainer): 시각화에 사용할 CAMExplainer 인스턴스.
        face (np.ndarray): 전처리(크롭)가 완료된 얼굴 이미지 배열 (RGB).
        category (int): 타겟 클래스 인덱스 (위조: 1, 정상: 0).
        explain_req_dict (dict): 시각화 요청 파라미터 딕셔너리. (_run_visualization과 동일)

    Returns:
        np.ndarray: 시각화가 적용된 얼굴 이미지 (RGB, np.uint8).
    """
    if explain_req_dict["display_type"] == "heatmap":
        return explainer.display_heatmap_from_array(face, image_weight=explain_req_dict["overlay_ratio"], threshold=explain_req_dict["threshold"],
                                                 category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])
    elif explain_req_dict["display_type"] == "bbox":  
        return explainer.display_bbox_from_array(face, threshold=explain_req_dict["threshold"],
                                              category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"]) 
    else: # display_type == "heatmap_bbox"
        return explainer.display_heatmap_bbox_from_array(face, image_weight=explain_req_dict["overlay_ratio"], threshold=explain_req_dict["threshold"], 
                                                       category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])

@celery_app.task(name="process_explain_image_task")
def process_explain_image_task(user_email: str, 
                               version_type: str,
                               domain_type: str,
                               image_loc: str,
                               image_id: int,
                               category: int,
                               explain_req_dict: dict) -> dict:
    """딥페이크 이미지 위조 흔적 시각화를 처리하는 Celery 비동기 태스크.

    CAMExplainer로 시각화 이미지를 생성하고 서버에 저장한다.
    Celery 워커(동기) 내에서 asyncio 이벤트 루프를 직접 구동해 비동기 로직을 실행한다.

    Args:
        user_email (str): 요청 사용자 이메일 (저장 경로 구분용).
        version_type (str): 모델 버전 타입 (MODEL_CONFIG 키값).
        domain_type (str): 탐지 도메인 타입 (MODEL_CONFIG 키값).
        image_loc (str): 분석 대상 이미지의 서버 저장 경로 (DB 기준, '/'로 시작).
        image_id (int): 이미지 레코드 PK.
        category (int): 타겟 클래스 인덱스 (위조: 1, 정상: 0).
        explain_req_dict (dict): 시각화 요청 파라미터 딕셔너리.

    Returns:
        dict:
            - status (str): "SUCCESS" | "FAILED"
            - message (str): 처리 결과 메세지.
            - cam_loc (str | None): 저장된 시각화 이미지 경로. 실패 시 None.

    Note:
        생성된 시각화 파일은 태스크 완료 60초 후 자동 삭제된다.
    """
    async def run_explain():
        cam_loc = None
        try:
            model_name, dataset = inference_svc.MODEL_CONFIG[version_type][explain_req_dict["model_type"]][domain_type]
        
            explainer = _get_or_create_explainer(model_name, dataset, explain_req_dict)
            image_path = "." + image_loc

            # 시각화 이미지 생성 시작
            try:
                image = _run_visualization(explainer, image_path, category, explain_req_dict)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="딥페이크 이미지 위조 흔적을 생성하는 중 오류가 발생하였습니다"
                )

            # 생성된 시각화 이미지 파일 저장 
            try:
                cam_loc = await image_svc.upload_image_cam(user_email, image_id, image)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="딥페이크 이미지 위조 흔적 파일을 저장하는 중 오류가 발생했습니다."
                )
            
            return {"status": "SUCCESS", 
                    "message": "딥페이크 이미지 위조 흔적 시각화가 성공적으로 이루어졌습니다",
                    "cam_loc": cam_loc}
        
        except HTTPException as e:
            print(e.detail)
            return {"status": "FAILED", "message": str(e.detail)}
        
        except Exception as e:
            print(str(e))                    
            return {"status": "FAILED", "message": str(e)}
        
        finally:
            # 임시 저장된 시각화 이미지 파일 삭제 
            if cam_loc:
                image_svc.cleanup_image_cam.apply_async(args=[cam_loc], countdown=60) 

    # 동기식 Celery 워커 내 비동기 이벤트 루프 구동
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_explain())
    finally:
        loop.close()
        
@celery_app.task(name="process_explain_frame_task")
def process_explain_frame_task(user_email: str, 
                               version_type: str,
                               domain_type: str,
                               video_loc: str,
                               video_id: int,
                               category: int,
                               frame_time: float, 
                               explain_req_dict: dict) -> dict:
    """딥페이크 비디오 특정 프레임의 위조 흔적 시각화를 처리하는 Celery 비동기 태스크.

    비디오에서 프레임을 추출하고 얼굴을 크롭한 뒤 CAMExplainer로 시각화 이미지를 생성해 저장한다.
    Celery 워커(동기) 내에서 asyncio 이벤트 루프를 직접 구동해 비동기 로직을 실행한다.

    Args:
        user_email (str): 요청 사용자 이메일 (저장 경로 구분용).
        version_type (str): 모델 버전 타입 (MODEL_CONFIG 키값).
        domain_type (str): 탐지 도메인 타입 (MODEL_CONFIG 키값).
        video_loc (str): 분석 대상 비디오의 서버 저장 경로 (DB 기준, '/'로 시작).
        video_id (int): 비디오 레코드 PK.
        category (int): 타겟 클래스 인덱스 (위조: 1, 정상: 0).
        frame_time (float): 시각화할 프레임의 타임스탬프 (초 단위).
        explain_req_dict (dict): 시각화 요청 파라미터 딕셔너리.

    Returns:
        dict:
            - status (str): "SUCCESS" | "FAILED"
            - message (str): 처리 결과 메세지.
            - cam_loc (str | None): 저장된 시각화 이미지 경로. 실패 시 None.

    Note:
        생성된 시각화 파일은 태스크 완료 60초 후 자동 삭제된다.
    """
    async def run_explain():
        cam_loc = None
        try:
            model_name, dataset = inference_svc.MODEL_CONFIG[version_type][explain_req_dict["model_type"]][domain_type]
            video_path = "." + video_loc

            explainer = _get_or_create_explainer(model_name, dataset, explain_req_dict)

            face = _extract_face_from_frame(video_path, frame_time, explainer)
            
            # 비디오 프레임 시각화 생성 시작
            try:
                image = _run_visualization_from_array(explainer, face, category, explain_req_dict)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="딥페이크 비디오 프레임 위조 흔적을 생성하는 중 오류가 발생하였습니다"
                )

            # 비디오 프레임 시각화 파일 저장 
            try:
                cam_loc = await image_svc.upload_frame_cam(user_email, video_id, frame_time, image)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="딥페이크 비디오 프레임 위조 흔적 파일을 저장하는 중 오류가 발생했습니다."
                )
            
            return {"status": "SUCCESS", 
                    "message": "딥페이크 비디오 프레임 위조 흔적 시각화가 성공적으로 이루어졌습니다",
                    "cam_loc": cam_loc}
        
        except HTTPException as e:
            print(e.detail)
            return {"status": "FAILED", "message": str(e.detail)}
        
        except Exception as e:
            print(str(e))                    
            return {"status": "FAILED", "message": str(e)}
        
        finally:
            # 임시 저장된 비디오 프레임 시각화 파일 삭제 
            if cam_loc:
                image_svc.cleanup_image_cam.apply_async(args=[cam_loc], countdown=60) 

    # 동기식 Celery 워커 내 비동기 이벤트 루프 구동
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_explain())
    finally:
        loop.close()

