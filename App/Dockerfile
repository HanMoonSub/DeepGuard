# ===== Common Base (CPU only) =====
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies (CPU-only torch, no CUDA bundle)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy project structure
COPY deepguard/ ./deepguard/
COPY inference/ ./inference/
COPY explainability/ ./explainability/
COPY preprocess/ ./preprocess/
COPY App/ ./App/
COPY pyproject.toml README.md ./

# deepguard/inference/explainability/preprocess를 site-packages로 설치
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

WORKDIR /app/App


# ===== API Stage (FastAPI) =====
FROM base AS fastapi

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# ===== Worker Stage (Celery) =====
FROM base AS celery

# Default concurrency
ENV CELERY_CONCURRENCY=2

CMD celery -A celery_app worker --loglevel=info --concurrency=$CELERY_CONCURRENCY
