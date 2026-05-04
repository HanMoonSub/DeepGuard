import os
import sys
from pathlib import Path
from celery import Celery
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

# 기본값 설정
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", 6379)

REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# Celery Instance 생성
celery_app = Celery(
    "deepguard",
    broker=REDIS_URL, # 메세지를 전달할 브로커
    backend=REDIS_URL, # 작업 결과를 저장할 백엔드
    include=["services.inference_svc"] # 비동기 추론 작업 파일
)

# Celery Instance 상세 설정
celery_app.conf.update(
    # 데이터 직렬화 방식
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # 시간대 설정
    timezone='Asia/Seoul',
    enable_utc=True,
    
    # GPU 메모리 오버플로우 방지
    worker_prefetch_multiplier=1,
    
    # 작업 도중 Worker 종료 시, 다시 큐에 넣는다
    task_acks_late=True,
    
    # 작업 결과 보관 시간 (초 단위)
    result_expires=3600
)