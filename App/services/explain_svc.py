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
    ClipHiResCAMExplainer, ClipGradCAMElementWiseExplainer, ClipLayerCAMExplainer,
    ClipEigenGradCAMExplainer, ClipGradCAMPlusPlusExplainer, ClipXGradCAMExplainer,
    BranchFusionExplainer, ClipBranchFusionExplainer,
)
from celery_app import celery_app

# LOW branch: captures local forgery artifacts / HIGH branch: captures global forgery artifacts
EXPLAINER_REGISTRY = {
    "hirescam":   HiResCAMExplainer, # LOW branch
    "gradcamelementwise": GradCAMElementWiseExplainer, # LOW branch
    "layercam":   LayerCAMExplainer, # LOW branch
    "eigengradcam": EigenGradCAMExplainer, # HIGH branch
    "gradcamplusplus":   GradCAMPlusPlusExplainer, # HIGH branch
    "xgradcam":     XGradCAMExplainer, # HIGH branch
}

# both (combined branches) settings — cannot be changed by the client
FUSION_MODE = "mean"
FUSION_LOW_WEIGHT = 0.6

# Explainers for v2 (DeepClip)
CLIP_EXPLAINER_REGISTRY = {
    "hirescam":   ClipHiResCAMExplainer, # LOW branch
    "gradcamelementwise": ClipGradCAMElementWiseExplainer, # LOW branch
    "layercam":   ClipLayerCAMExplainer, # LOW branch
    "eigengradcam": ClipEigenGradCAMExplainer, # HIGH branch
    "gradcamplusplus":   ClipGradCAMPlusPlusExplainer, # HIGH branch
    "xgradcam":     ClipXGradCAMExplainer, # HIGH branch
}

_explainer_cache: dict = {}

def _extract_face_from_frame(video_path: str, frame_time: float, explainer: CAMExplainer | BranchFusionExplainer) -> np.ndarray:
    """Extract the face region from the frame at a given time in a video and return the cropped array.

    Args:
        video_path (str): Video file path.
        frame_time (float): Time of the frame to extract (in seconds).
        explainer (CAMExplainer): CAMExplainer instance used for face detection and cropping.

    Returns:
        np.ndarray: Cropped face image array (RGB).
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
    ret, frame = cap.read()
    cap.release()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # BranchFusionExplainer delegates face detection/cropping to the low explainer
    detector = getattr(explainer, "low_explainer", explainer)
    bbox = detector._get_face_bbox(frame_rgb)

    return detector._crop_face(frame_rgb, bbox[:4])

def _get_or_create_explainer(model_name: str, dataset: str | None, explain_req_dict: dict) -> CAMExplainer | BranchFusionExplainer:
    """Get an Explainer from the cache, or create, cache and return a new one if missing.

    The cache key consists of 4 parameters: (model_name, dataset, explainer_type, branch_level).
    Inference-time parameters (aug_smooth, overlay_ratio, etc.) are left out of the key to minimize memory use.

    Args:
        model_name (str): Model name used for inference. deepclip_* uses a Clip explainer.
        dataset (str | None): Dataset the model was trained on. None for DeepClip.
        explain_req_dict (dict): Visualization request parameters.
            - explainer_type (str): EXPLAINER_REGISTRY key ("fusion" for both).
            - branch_level (str): Model branch level ("low" | "high" | "both").

    Returns:
        CAMExplainer | BranchFusionExplainer: Cached or newly created Explainer instance.
    """
    cache_key = (model_name, dataset, explain_req_dict["explainer_type"], explain_req_dict["branch_level"])
    is_clip = model_name.startswith("deepclip")

    if cache_key not in _explainer_cache:
        if explain_req_dict["branch_level"] == "both":
            # both: combine low (LayerCAM) + high (EigenGradCAM) CAMs with a weighted mean (fuse/low_weight are fixed on the server)
            if is_clip:
                _explainer_cache[cache_key] = ClipBranchFusionExplainer(
                    model_name = model_name,
                    low = ClipLayerCAMExplainer,
                    high = ClipEigenGradCAMExplainer,
                    fuse = FUSION_MODE,
                    low_weight = FUSION_LOW_WEIGHT,
                )
            else:
                _explainer_cache[cache_key] = BranchFusionExplainer(
                    model_name = model_name,
                    dataset = dataset,
                    low = LayerCAMExplainer,
                    high = EigenGradCAMExplainer,
                    fuse = FUSION_MODE,
                    low_weight = FUSION_LOW_WEIGHT,
                )
        else:
            registry = CLIP_EXPLAINER_REGISTRY if is_clip else EXPLAINER_REGISTRY
            _explainer_cache[cache_key] = registry[explain_req_dict["explainer_type"]](
                model_name = model_name,
                dataset = dataset,
                branch_level = explain_req_dict["branch_level"],
            )
    return _explainer_cache[cache_key]
        
def _run_visualization(explainer: CAMExplainer, image_path: str, category: int, explain_req_dict: dict) -> np.ndarray:
    """Take an image path and generate a visualization image according to display_type.

    Args:
        explainer (CAMExplainer): CAMExplainer instance used for visualization.
        image_path (str): Input image file path.
        category (int): Target class index (fake: 1, real: 0).
        explain_req_dict (dict): Visualization request parameters.
            - display_type (str): "heatmap" | "bbox" | "heatmap_bbox"
            - overlay_ratio (float): Weight of the original image when overlaying the heatmap.
            - aug_smooth (bool): Whether to apply test-time augmentation.
            - eigen_smooth (bool): Whether to apply PCA-based noise removal.

    Returns:
        np.ndarray: Face image with the visualization applied (RGB, np.uint8).
    """
    if explain_req_dict["display_type"] == "heatmap":
        return explainer.display_heatmap_on_image(image_path, image_weight=explain_req_dict["overlay_ratio"],
                                                 category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])
    elif explain_req_dict["display_type"] == "bbox":  
        return explainer.display_bbox_on_image(image_path,
                                              category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"]) 
    else: # display_type == "heatmap_bbox"
        return explainer.display_heatmap_bbox_on_image(image_path, image_weight=explain_req_dict["overlay_ratio"], 
                                                       category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])

def _run_visualization_from_array(explainer: CAMExplainer, face: np.ndarray, category: int, explain_req_dict: dict) -> np.ndarray:
    """Take a cropped face array and generate a visualization image according to display_type.

    Args:
        explainer (CAMExplainer): CAMExplainer instance used for visualization.
        face (np.ndarray): Preprocessed (cropped) face image array (RGB).
        category (int): Target class index (fake: 1, real: 0).
        explain_req_dict (dict): Visualization request parameters (same as _run_visualization).

    Returns:
        np.ndarray: Face image with the visualization applied (RGB, np.uint8).
    """
    if explain_req_dict["display_type"] == "heatmap":
        return explainer.display_heatmap_from_array(face, image_weight=explain_req_dict["overlay_ratio"],
                                                 category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])
    elif explain_req_dict["display_type"] == "bbox":  
        return explainer.display_bbox_from_array(face,
                                              category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"]) 
    else: # display_type == "heatmap_bbox"
        return explainer.display_heatmap_bbox_from_array(face, image_weight=explain_req_dict["overlay_ratio"], 
                                                       category=category, aug_smooth=explain_req_dict["aug_smooth"], eigen_smooth=explain_req_dict["eigen_smooth"])

@celery_app.task(name="process_explain_image_task")
def process_explain_image_task(user_email: str, 
                               version_type: str,
                               domain_type: str,
                               image_loc: str,
                               image_id: int,
                               category: int,
                               explain_req_dict: dict) -> dict:
    """Celery async task that visualizes forgery artifacts in a deepfake image.

    Generates the visualization with CAMExplainer and saves it on the server.
    Runs the async logic by driving an asyncio event loop directly inside the (synchronous) Celery worker.

    Args:
        user_email (str): Requesting user's email (used to separate storage paths).
        version_type (str): Model version type (MODEL_CONFIG key).
        domain_type (str): Detection domain type (MODEL_CONFIG key).
        image_loc (str): Stored server path of the target image (DB format, starts with '/').
        image_id (int): Image record PK.
        category (int): Target class index (fake: 1, real: 0).
        explain_req_dict (dict): Visualization request parameters.

    Returns:
        dict:
            - status (str): "SUCCESS" | "FAILED"
            - message (str): Result message.
            - cam_loc (str | None): Path of the saved visualization image, or None on failure.

    Note:
        The generated visualization file is deleted automatically 60 seconds after the task finishes.
    """
    async def run_explain():
        cam_loc = None
        try:
            model_name, dataset = inference_svc.MODEL_CONFIG[version_type][explain_req_dict["model_type"]][domain_type]
        
            explainer = _get_or_create_explainer(model_name, dataset, explain_req_dict)
            image_path = "." + image_loc

            # Start generating the visualization image
            try:
                image = _run_visualization(explainer, image_path, category, explain_req_dict)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="딥페이크 이미지 위조 흔적을 생성하는 중 오류가 발생하였습니다"
                )

            # Save the generated visualization image file
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
            # Delete the temporarily stored visualization image file
            if cam_loc:
                image_svc.cleanup_image_cam.apply_async(args=[cam_loc], countdown=60) 

    # Drive an async event loop inside the synchronous Celery worker
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
    """Celery async task that visualizes forgery artifacts in a specific frame of a deepfake video.

    Extracts the frame from the video, crops the face, then generates and saves the visualization with CAMExplainer.
    Runs the async logic by driving an asyncio event loop directly inside the (synchronous) Celery worker.

    Args:
        user_email (str): Requesting user's email (used to separate storage paths).
        version_type (str): Model version type (MODEL_CONFIG key).
        domain_type (str): Detection domain type (MODEL_CONFIG key).
        video_loc (str): Stored server path of the target video (DB format, starts with '/').
        video_id (int): Video record PK.
        category (int): Target class index (fake: 1, real: 0).
        frame_time (float): Timestamp of the frame to visualize (in seconds).
        explain_req_dict (dict): Visualization request parameters.

    Returns:
        dict:
            - status (str): "SUCCESS" | "FAILED"
            - message (str): Result message.
            - cam_loc (str | None): Path of the saved visualization image, or None on failure.

    Note:
        The generated visualization file is deleted automatically 60 seconds after the task finishes.
    """
    async def run_explain():
        cam_loc = None
        try:
            model_name, dataset = inference_svc.MODEL_CONFIG[version_type][explain_req_dict["model_type"]][domain_type]
            video_path = "." + video_loc

            explainer = _get_or_create_explainer(model_name, dataset, explain_req_dict)

            face = _extract_face_from_frame(video_path, frame_time, explainer)
            
            # Start generating the video frame visualization
            try:
                image = _run_visualization_from_array(explainer, face, category, explain_req_dict)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="딥페이크 비디오 프레임 위조 흔적을 생성하는 중 오류가 발생하였습니다"
                )

            # Save the video frame visualization file
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
            # Delete the temporarily stored video frame visualization file
            if cam_loc:
                image_svc.cleanup_image_cam.apply_async(args=[cam_loc], countdown=60) 

    # Drive an async event loop inside the synchronous Celery worker
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_explain())
    finally:
        loop.close()

