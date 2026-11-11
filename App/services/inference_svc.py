import asyncio
from typing import Literal
from fastapi import status, Depends
from fastapi.exceptions import HTTPException
from sqlalchemy import text, Connection
from sqlalchemy.exc import SQLAlchemyError
from services import image_svc, video_svc
from db.database import celery_db_conn
from celery_app import celery_app
from inference.image_predictor_prt import ImagePredictor, ClipImagePredictor
from inference.video_predictor_prt import VideoPredictor, ClipVideoPredictor
from inference.utils import PredictorError

# Verdicts use only v1 (MS-EffGCViT, per-dataset weights)
MODEL_CONFIG = {
    'v1': {
        'fast': {
            '서양인': ("ms_eff_gcvit_b0", "ff++"),
            '동양인': ("ms_eff_gcvit_b0", "kodf")
        },
        'pro': {
            '서양인': ("ms_eff_gcvit_b5", "ff++"),
            '동양인': ("ms_eff_gcvit_b5", "kodf")
        }
    }
}

# DeepClip: on-demand analysis of the 20 text (region x type) probabilities for FAKE results (no dataset, services/artifact_svc.py)
DEEPCLIP_MODEL = {'fast': 'deepclip_b0', 'pro': 'deepclip_b5'}

_image_cache: dict = {}
_video_cache: dict = {}

def _get_or_create_predictor(
    model_name: str,
    dataset: str | None,
    predictor_mode: Literal["image", "video"],
) -> ImagePredictor | VideoPredictor:
    """Get a Predictor from the cache, or create, cache and return a new one if missing.

    Image and video caches are kept separately so each mode has its own instances.

    Args:
        model_name (str): Model architecture name used for inference.
        dataset (str | None): Dataset the model was trained on (None for DeepClip).
        predictor_mode (Literal["image", "video"]): Which Predictor to use.

    Returns:
        ImagePredictor | VideoPredictor: Cached or newly created Predictor instance.
    """
    cache_key = (model_name, dataset)
    is_clip = model_name.startswith("deepclip") # DeepClip takes no dataset argument

    if predictor_mode == "image":
        if cache_key not in _image_cache:
            if is_clip:
                _image_cache[cache_key] = ClipImagePredictor(margin_ratio=0.2, conf_thres=0.5, model_name=model_name)
            else:
                _image_cache[cache_key] = ImagePredictor(
                    margin_ratio=0.2,
                    conf_thres=0.5,
                    model_name=model_name,
                    dataset=dataset,
                )
        return _image_cache[cache_key]

    if cache_key not in _video_cache:
        if is_clip:
            _video_cache[cache_key] = ClipVideoPredictor(margin_ratio=0.2, conf_thres=0.5, model_name=model_name)
        else:
            _video_cache[cache_key] = VideoPredictor(
                margin_ratio=0.2,
                conf_thres=0.5,
                model_name=model_name,
                dataset=dataset,
            )
    return _video_cache[cache_key]

def predict_image(image_loc: str, version_type: str, model_type: str, domain_type: str) -> dict:
    """Take an image path, infer whether it is a deepfake, and return the analysis result.

    Args:
        image_loc (str): Stored server path of the target image (DB format, starts with '/').
        version_type (str): Model version type (MODEL_CONFIG key).
        model_type (str): Inference mode ("fast" | "pro").
        domain_type (str): Detection domain ("서양인" | "동양인").

    Returns:
        dict:
            - analysis (dict): Includes prob.
              All values are -1 when no face is detected or an error occurs.
            - message (str): Result message.
            - status (str): "success" | "warning" | "failed"
    """
    
    model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]

    predictor = _get_or_create_predictor(model_name, dataset, "image")
    image_path = "." + image_loc
    
    try:
        analysis = predictor.predict_img(image_path)
        
        print(f"딥페이크 확률 값: {analysis['prob']}")
    
        # On successful analysis
        return {
            "analysis": analysis,
            "message": "이미지 분석에 성공하였습니다",
            "status": "success"
        }
    except PredictorError as e:
        print(e.message)
        return {
            "analysis": {"prob": -1},
            "message": e.message,
            "status": "warning",
        }
    
    # CUDA out of memory, CUDA device error, timm model loading failure, etc.
    except Exception as e:
        print(str(e))
        return {
            "analysis": {"prob": -1},
            "message": f"이미지 분석 중 치명적인 오류가 발생했습니다.",
            "status": "failed"
        }

@celery_app.task(name="process_image_task")
def process_image_task(image_id: int, image_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None) -> None:
    """Celery async task that runs deepfake image inference.

    Saves the inference result to the DB, and marks the record as failed if a SQLAlchemy error occurs.
    Runs the async logic by driving an asyncio event loop directly inside the (synchronous) Celery worker.

    Args:
        image_id (int): Image record PK.
        image_loc (str): Stored server path of the target image (DB format, starts with '/').
        version_type (str): Model version type (MODEL_CONFIG key).
        model_type (str): Inference mode ("fast" | "pro").
        domain_type (str): Detection domain ("서양인" | "동양인").
        user_id (int | None): Logged-in user ID, or None for guests.

    Note:
        For guest requests (user_id=None), the image file and DB record are deleted automatically 60 seconds after inference finishes.
    """
    async def run_inference():    
        try:
            # Run image inference and return the deepfake result
            result = predict_image(image_loc, version_type, model_type, domain_type)
        
            async with celery_db_conn() as conn:    
                # Save the image inference result to the DB regardless of login
                await image_svc.update_image_result(
                    conn, 
                    image_id, 
                    result["analysis"], 
                    result["message"], 
                    result["status"]
                    )
        except SQLAlchemyError as e:
            print(f"[DB Error] Image Result Update Failed: {str(e)}")
            try:
                async with celery_db_conn() as conn:  
                    await image_svc.update_image_result(
                        conn, 
                        image_id, 
                        {"prob": -1}, 
                        "이미지 추론 결과 업데이트 중 오류가 발생하였습니다.", 
                        "failed"
                    )
            except Exception as db_err:
                print(f"Final Emergency DB Update Failed: {db_err}")
        
        except Exception as e:
            print(f"[Unknown Error] Image Task Failed: {str(e)}")
        
        finally:
            if not user_id:
                image_svc.cleanup_anonymous_image.apply_async(args=[image_id, image_loc], countdown=60)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_inference())
    finally:
        loop.close()
    
        
def predict_video(video_loc: str, version_type: str, model_type: str, domain_type: str) -> dict:
    """Take a video path, infer whether it is a deepfake, and return the analysis result.

    Args:
        video_loc (str): Stored server path of the target video (DB format, starts with '/').
        version_type (str): Model version type (MODEL_CONFIG key).
        model_type (str): Inference mode ("fast" | "pro").
        domain_type (str): Detection domain ("서양인" | "동양인").

    Returns:
        dict:
            - analysis (dict): Includes prob, frame_results, fps, total_frames, num_sampled, num_extracted, num_detected.
              Key numeric values are -1 when an error occurs.
            - message (str): Result message.
            - status (str): "success" | "warning" | "failed"
    """
    model_name, dataset = MODEL_CONFIG[version_type][model_type][domain_type]

    predictor = _get_or_create_predictor(model_name, dataset, "video")
    video_path = "." + video_loc
    
    try:
        analysis = predictor.predict_video(video_path)
        
        print(f"딥페이크 확률 값: {analysis['prob']}")
    
        # On successful analysis
        return {
            "analysis": analysis,
            "message": "비디오 분석에 성공하였습니다",
            "status": "success"
        }
    except PredictorError as e:
        print(e.message)
        return {
            "analysis": {"prob": -1, "frame_results": [],
                         "fps": None, "total_frames": None, "num_sampled": None, "num_extracted": None, "num_detected": None,},
            "message": e.message,
            "status": "warning",
        }
    
    # CUDA out of memory, CUDA device error, timm model loading failure, etc.
    except Exception as e:
        print(str(e))
        return {
            "analysis": {"prob": -1, "frame_results": [],
                         "fps": None, "total_frames": None, "num_sampled": None, "num_extracted": None, "num_detected": None,},
            "message": f"비디오 분석 중 치명적인 오류가 발생했습니다.",
            "status": "failed"
        }
        
@celery_app.task(name="process_video_task")
def process_video_task(video_id: int, video_loc: str, version_type: str, model_type: str, 
                             domain_type: str, user_id: int | None) -> None:
    """Celery async task that runs deepfake video inference.

    Saves the inference result to the DB and, on success, also saves the meta data and per-frame results.
    Failures saving the meta/frame data are handled independently and do not affect the main inference result.
    Runs the async logic by driving an asyncio event loop directly inside the (synchronous) Celery worker.

    Args:
        video_id (int): Video record PK.
        video_loc (str): Stored server path of the target video (DB format, starts with '/').
        version_type (str): Model version type (MODEL_CONFIG key).
        model_type (str): Inference mode ("fast" | "pro").
        domain_type (str): Detection domain ("서양인" | "동양인").
        user_id (int | None): Logged-in user ID, or None for guests.

    Note:
        For guest requests (user_id=None), the video file and DB record are deleted automatically 60 seconds after inference finishes.
    """
    async def run_inference():
        try:
            # Run video inference and return the deepfake result
            result = predict_video(video_loc, version_type, model_type, domain_type)
        
            async with celery_db_conn() as conn:    
                # Save the inference result to the DB regardless of login
                await video_svc.update_video_result(
                    conn, 
                    video_id, 
                    result["analysis"], 
                    result["message"], 
                    result["status"]
                    )
            if result["status"] == "success":
                if result["analysis"].get("fps") is not None:
                    try:
                        async with celery_db_conn() as conn:
                            await video_svc.save_video_meta_result(conn, video_id, result["analysis"])
                    except Exception as e:
                        print(f"[Video Meta Save Error] video_id={video_id}: {e}")

                if result["analysis"].get("frame_results"):
                    try:
                        async with celery_db_conn() as conn:
                            await video_svc.save_video_frame_result(
                                conn, video_id, result["analysis"]["frame_results"]
                            )
                    except Exception as e:
                        print(f"[Video Frame Save Error] video_id={video_id}: {e}")
                        
        except SQLAlchemyError as e:
            print(f"Video DB Update Error: {e}")
            try:
                async with celery_db_conn() as conn:  
                    await video_svc.update_video_result(
                        conn, 
                        video_id, 
                        {"prob": -1}, 
                        "비디오 추론 결과 업데이트 중 오류가 발생하였습니다.", 
                        "failed"
                    )          
            except Exception as db_err:
                print(f"Final Emergency DB Update Failed: {db_err}")
        
        except Exception as e:
            print(f"[Unknown Error] Video Task Failed: {str(e)}")
          
        finally:
            if not user_id:
                is_success = result.get("status") == "success"
                video_svc.cleanup_anonymous_video.apply_async(args=[video_id, video_loc, is_success], countdown=60)

                

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_inference())
    finally:
        loop.close()
    