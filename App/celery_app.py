import os
from celery import Celery
from dotenv import load_dotenv

# ------ Load .env file ------
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Create the Celery instance
celery_app = Celery(
    "deepguard",
    broker=REDIS_URL, # broker that delivers messages
    backend=REDIS_URL, # backend that stores task results
    # Async inference task modules
    include=["services.inference_svc", # process_image_task, process_video_task 
             "services.explain_svc", # process_explain_image_task
             "services.artifact_svc", # process_artifact_image_task, process_artifact_video_task
             "services.image_svc", # cleanup_anonymous_image, cleanup_image_cam
             "services.video_svc"] # cleanup_anonymous_video
)

# Detailed Celery instance settings
celery_app.conf.update(
    # Data serialization format
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Time zone settings
    timezone='Asia/Seoul',
    enable_utc=True,
    
    # Prevent GPU memory overflow
    worker_prefetch_multiplier=1,
    
    # Re-queue the task if the worker dies mid-task
    task_acks_late=True,
    
    # How long task results are kept (in seconds)
    result_expires=3600
)