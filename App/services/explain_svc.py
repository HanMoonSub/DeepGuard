import cv2
import asyncio
from functools import partial
from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import Connection
from schemas.explain_schema import ExplainRequest
from services import image_svc, inference_svc
from explainability import (
    HiResCAMExplainer, GradCAMElementWiseExplainer, LayerCAMExplainer,
    EigenGradCAMExplainer, GradCAMPlusPlusExplainer, XGradCAMExplainer,
)

EXPLAINER_REGISTRY = {
    # LOW branch: 국소 위조 흔적 포착 
    "hirescam":   HiResCAMExplainer,
    "gradcamelementwise": GradCAMElementWiseExplainer,
    "layercam":   LayerCAMExplainer,
    # HIGH branch: 전역 위조 흔적 포착
    "eigengradcam": EigenGradCAMExplainer,
    "gradcamplusplus":   GradCAMPlusPlusExplainer,
    "xgradcam":     XGradCAMExplainer,
}

_explainer_cache: dict = {}

def _get_or_create_explainer(model_name: str, dataset: str, explain_req: ExplainRequest):
    cache_key = (model_name, dataset,
                 explain_req.explainer_type, explain_req.branch_level,
                 explain_req.category, explain_req.aug_smooth, explain_req.eigen_smooth)
    
    if cache_key not in _explainer_cache:
        _explainer_cache[cache_key] = EXPLAINER_REGISTRY[explain_req.explainer_type](
            model_name = model_name,
            dataset = dataset,
            branch_level = explain_req.branch_level,
            category = explain_req.category,
            aug_smooth = explain_req.aug_smooth,
            eigen_smooth = explain_req.eigen_smooth,
        )
    return _explainer_cache[cache_key]
        
def _run_visualization(explainer, image_path: str, explain_req: ExplainRequest) -> bytes:
    if explain_req.display_type == "heatmap":
        img = explainer.display_heatmap_on_image(image_path, image_weight=explain_req.overlay_ratio)
    elif explain_req.display_type == "contour":
        img = explainer.display_contour_on_image(image_path)
    else:  
        img = explainer.display_bbox_on_image(image_path)
 
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes()

async def explain_image(conn: Connection, image_id: int, explain_req: ExplainRequest):
    result = await image_svc.get_image_result(conn, image_id)
    
    if result.status != "SUCCESS":
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "이미지 위조 흔적 분석은 추론이 성공한 이미지만 가능합니다"
        )
    
    model_name, dataset = inference_svc.MODEL_CONFIG[result.version_type][result.model_type][result.domain_type]
    
    explainer = _get_or_create_explainer(model_name, dataset, explain_req)
    image_path = "." + result.image_loc
    
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            partial(_run_visualization, explainer, image_path, explain_req)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="알수없는 이유로 문제가 발생하였습니다."
            )
