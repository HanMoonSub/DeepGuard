# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

DeepGuard is a deepfake detection research project built around two custom hybrid CNN-ViT
architectures (**MS-EffViT** and **MS-EffGCViT**). The repo has three largely independent parts
that share the same Python package but have separate dependency/build lifecycles:

- **`deepguard/`** — the pip-installable core package (models, layers, data pipeline, optimizer,
  scheduler, training utils). This is what gets published to PyPI as `deepguard`.
- **`inference/`**, **`explainability/`**, **`preprocess/`** — also part of the published package
  (see `[tool.setuptools.packages.find]` in `pyproject.toml`), providing prediction and Grad-CAM-style
  XAI on top of `deepguard` models.
- **`App/`** + **`client/`** — a separate full-stack web app (FastAPI backend + React frontend)
  that serves the models via a REST API. These are explicitly *excluded* from the PyPI package
  (`exclude = ["App*", "client*"]` in `pyproject.toml`).

Top-level `train_eff_vit.py` / `train_eff_gcvit.py` are standalone training entrypoints, not part
of the installable package.

## Commands

### Python package (deepguard / inference / explainability / preprocess)

```bash
pip install -r requirements.txt      # full dev environment (backend + ML + lint)
ruff check .                         # lint
ruff format --check .                # format check (CI uses --check; drop it to auto-format)
pytest --tb=short -q                 # run tests (mirrors PR CI)
pytest path/to/test_file.py::test_name   # run a single test
```

CI (`.github/workflows/pr-ci.yml`) runs `ruff check`, `ruff format --check`, then `pytest` on every
PR into `main`/`develop`, on Python 3.11. There is no `ruff` config section in `pyproject.toml`, so
default ruff rules apply.

### Training

```bash
python -m train_eff_vit --root-dir DATA_ROOT --model-ver ms_eff_vit_b5 --dataset ff++ --seed 2025 --wandb-api-key <key>
python -m train_eff_gcvit --root-dir DATA_ROOT --model-ver ms_eff_gcvit_b0 --dataset celeb_df_v2 --seed 2025 --wandb-api-key <key>
```

`--model-ver` ∈ `{ms_eff_vit_b0, ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5}`,
`--dataset` ∈ `{ff++, celeb_df_v2, kodf}`. Per-model/per-dataset hyperparameters live in
`deepguard/config/<model_ver>/<dataset>.yaml` (OmegaConf) and training is tracked via
Weights & Biases (`wandb`).

### Inference / evaluation

```bash
python -m inference.predict_video --root-dir DATA_ROOT --model-name ms_eff_gcvit_b0 --model-dataset kodf --num-frames 20 --agg-mode conf
```

### Backend (App/, FastAPI)

Run from inside `App/` (routes import siblings like `from routes import ...`, `from services import ...`
directly, not as a package):

```bash
cd App
uvicorn main:app --reload            # dev server, expects Uvicorn's default port 8000
celery -A celery_app worker --loglevel=info   # background worker for async inference/explain tasks
```

Requires a `.env` in `App/` (see `App/.env` keys: `DB_*` for MySQL, `UPLOAD_DIR`/`EXPLAIN_UPLOAD_DIR`,
`HF_TOKEN`, `REDIS_HOST`/`REDIS_PORT`, `CORS_ALLOWED_ORIGINS`) and a running MySQL + Redis instance.
DB schema/seed SQL lives in `App/sql/`.

### Frontend (client/, React via Create React App)

```bash
cd client
npm install
npm start      # dev server on :3000, proxies /api and /static/{uploads,explain} to http://127.0.0.1:8000 (see src/setupProxy.js)
npm test
npm run build
```

## Architecture

### Model design (`deepguard/`)

Both `MultiScaleEffViT` (`deepguard/models/ms_eff_vit.py`) and `MultiScaleEffGCViT`
(`deepguard/models/ms_eff_gcvit.py`) follow the same dual-branch pattern, registered with
`timm` via `register_model`/`build_model_with_cfg` so they're usable through
`timm.create_model(...)` as well as direct import:

1. **`FeatExtractor`** (`deepguard/layers/featextractor.py`) wraps a `timm` EfficientNet backbone
   in `features_only=True` mode and taps two intermediate feature maps: a **low-level / high-resolution**
   one (`l_block_idx` ∈ {0,1,2}) for local forgery artifacts (texture, blending, compression traces),
   and a **high-level / low-resolution** one (`h_block_idx` ∈ {4,6}) for global semantic structure
   (lighting, geometry, shadows).
2. Each tapped feature map feeds its own small transformer branch — plain window attention for
   MS-EffViT, or GCViT-style local+global window attention (`deepguard/layers/window.py`,
   `global_query.py`, `attention.py`) for MS-EffGCViT — configured independently via `l_*`/`h_*`
   prefixed constructor args (dims, depths, window sizes, heads, mlp ratio, dropout).
3. Outputs from both branches are combined into a single binary logit (`num_classes=1`,
   real=0/fake=1).

Pretrained weights are fetched per `(model_name, dataset)` pair from GitHub Releases URLs declared
in each model file's `weight_registry` dict (three dataset variants: `celeb_df_v2`, `ff++`, `kodf`).
Mirror weights are also published to Hugging Face Hub (`KoreaPeter/ms-eff-gcvit-deepfake`).

This low/high branch split is the central concept of the whole codebase — it also drives the XAI
design (below), the data augmentation choices, and the CLI flags across training/inference.

### Explainability (`explainability/`)

CAM-based explainers are assigned per branch based on empirical performance, not interchangeable:
- **low-level branch**: `HiResCAM`, `GradCAMElementWise`, `LayerCAM` (favor precise local activation)
- **high-level branch**: `EigenGradCAM`, `GradCAM++`, `XGradCAM` (favor cleaner class-discriminative
  global maps)

`explainability/explainer/base_explainer.py` defines the shared interface
(`display_heatmap_on_image`, `display_bbox_on_image`, `display_heatmap_bbox_on_image`); concrete
explainers in `cam_explainer.py`, `eigenvalue_explainer.py`, `gradient_explainer.py`,
`misc_explainer.py`, `perturbation_explainer.py` subclass it. `branch_level` (`"low"`/`"high"`) is
a required constructor arg that determines which backbone feature stage the hooks attach to.
`explainability/metrics/` (`road.py`, `perturbation_confidence.py`, `cam_mult_image.py`) implement
CAM-quality metrics used by `explainability/cam_evaluator.py`.

### Preprocessing (`preprocess/`)

Two-stage pipeline shared conceptually across all three datasets (`celeb_df_v2/`, `ff++/`, `kodf/`
each have their own `detect_original_face.py`, `crop_face.py`, `split_data.py`, `utils.py`, since
each raw dataset has different file layouts/metadata):

1. **Face detection on real videos only** (`preprocess/face_detector.py`, using `yolov8n-face.pt`
   at repo root) — fake videos in these benchmarks reuse the same face bounding boxes as their
   source real video, cutting detection workload ~80%. Frames are dynamically rescaled based on
   resolution before detection for consistent inference speed.
2. **Face cropping + landmark extraction** (`preprocess/landmark_detector.py` + per-dataset
   `crop_face.py`) reuses those boxes for both real and fake videos, adds a jittered margin, and
   detects 5 facial landmarks (saved as `.npy`) to support landmark-based augmentation
   (`deepguard/data/dataset_cutout.py`).

Output layout is `DATA_ROOT/{crops,landmarks}/{video_id}/*.{png,npy}` plus a
`train_frame_metadata.csv` — this is the layout every training/inference entrypoint's
`--root-dir` expects.

### Data pipeline (`deepguard/data/`)

`dataset.py` is the base `Dataset`; `dataset_cutmix.py`, `dataset_cutout.py`, `dataset_mixup.py`
are augmentation-specific subclasses selected at training time via config
(`mixup_alpha`/`mixup_prob`, `cutout_prob` in the YAML configs). `handle_imbalance.py` addresses
the real/fake class imbalance inherent to these benchmarks (e.g. Celeb-DF-v2 is ~86% fake).
`split_data.py` produces train/val splits; `transforms.py` holds Albumentations pipelines.

### Web app (`App/` + `client/`)

FastAPI backend follows router → service → db layering:
- `App/routes/*.py` — thin route handlers (`auth`, `home`, `image`, `video`, `inference`, `explain`),
  mounted under `/api` via `App/routes/__init__.py`'s `api_router`.
- `App/services/*.py` — business logic, DB access, and file I/O per domain; several functions
  (e.g. in `image_svc.py`, `video_svc.py`, `explain_svc.py`) are Celery tasks for async
  inference/explain/cleanup work, dispatched through `App/celery_app.py` (Redis-backed broker
  and result backend).
- `App/db/database.py` — two SQLAlchemy async engines: one pooled (`engine`, used by request-scoped
  `context_get_conn` dependency) and one unpooled (`celery_engine`/`celery_db_conn`, used inside
  Celery tasks since they run outside the FastAPI request lifecycle).
- `App/schemas/*.py` — Pydantic request/response models per domain.
- Session auth is cookie-based via Redis (`App/utils/middleware.py`'s `RedisSessionMiddleware`,
  checked in routes via `services.session_svc`), not JWT.
- Uploaded/generated images, videos, and CAM overlays are served from `/static` (mounted to the
  `static/` dir) — locations configured via `UPLOAD_DIR`/`EXPLAIN_UPLOAD_DIR` env vars.

React frontend (Create React App, no router state library beyond `react-router-dom`) lives under
`client/src/pages/` — one file per page (`mainpage`, `loginpage`, `signuppage`, `analysispage`,
`AnalysisDetailPage`, `VideoAnalysisPage`, `VideoTimelinePage`, `HeatmapPage`, `ImageHeatmapPage`).
`client/src/setupProxy.js` proxies `/api` and `/static/{uploads,explain}` to the FastAPI dev server
at `127.0.0.1:8000` — the backend must be running for `npm start` to work end-to-end.

## Conventions worth knowing

- Model/layer files mix English docstrings with Korean inline comments (`App/` especially is
  Korean-first); match existing language when editing a file rather than converting wholesale.
- Model constructor args consistently use `l_`/`h_` prefixes to mean low-level-branch vs
  high-level-branch — keep this naming when adding new dual-branch parameters.
- `dataset` values are always one of the literal strings `celeb_df_v2`, `ff++`, `kodf`; `model_name`
  values are always one of `ms_eff_vit_b0`, `ms_eff_vit_b5`, `ms_eff_gcvit_b0`, `ms_eff_gcvit_b5`
  — these show up as string args across training, inference, and explainability APIs.
