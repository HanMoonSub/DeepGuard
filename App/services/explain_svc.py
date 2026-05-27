import cv2
import asyncio
import numpy as np
from functools import partial
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

# --- CAM Explainer Registry ---
# LOW branch: 국소 위조 흔적 포착
# HIGH branch: 전역 위조 흔적 포착 
EXPLAINER_REGISTRY = {
    "hirescam":   HiResCAMExplainer, #LOW branch
    "gradcamelementwise": GradCAMElementWiseExplainer, #LOW branch
    "layercam":   LayerCAMExplainer, #LOW branch
    "eigengradcam": EigenGradCAMExplainer, # HIGH branch
    "gradcamplusplus":   GradCAMPlusPlusExplainer, # HIGH branch
    "xgradcam":     XGradCAMExplainer, # HIGH branch
}

_explainer_cache: dict = {}

# 캐시된 CAMExplainer 객체 반환하거나 새로 생성 
def _get_or_create_explainer(model_name: str, dataset: str, explain_req_dict: dict):
    cache_key = (model_name, dataset, explain_req_dict["explainer_type"], explain_req_dict["branch_level"])
    
    if cache_key not in _explainer_cache:
        _explainer_cache[cache_key] = EXPLAINER_REGISTRY[explain_req_dict["explainer_type"]](
            model_name = model_name,
            dataset = dataset,
            branch_level = explain_req_dict["branch_level"],
        )
    return _explainer_cache[cache_key]
        
# 시각화 이미지 생성 (heatmap, contour, bbox 선택)
def _run_visualization(explainer: CAMExplainer, image_path: str, category: int, explain_req_dict: dict) -> np.ndarray:
    if explain_req_dict["display_type"] == "heatmap":
        return explainer.display_heatmap_on_image(image_path, image_weight=explain_req_dict["overlay_ratio"], threshold=explain_req_dict["threshold"],
                                                 category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])
    elif explain_req_dict["display_type"] == "bbox":  
        return explainer.display_bbox_on_image(image_path, threshold=explain_req_dict["threshold"],
                                              category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"]) 
    else: # display_type == "heatmap_bbox"
        return explainer.display_heatmap_bbox_on_image(image_path, image_weight=explain_req_dict["overlay_ratio"], threshold=explain_req_dict["threshold"], 
                                                       category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])
# 딥페이크 이미지 위조 흔적 시각화 처리
@celery_app.task(name="process_explain_image_task")
def process_explain_image_task(user_email: str, 
                               version_type: str,
                               domain_type: str,
                               image_loc: str,
                               image_id: int,
                               category: int,
                               explain_req_dict: dict):
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
