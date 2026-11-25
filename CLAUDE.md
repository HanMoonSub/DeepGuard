# CLAUDE.md

# AI Policy — MANDATORY

Read `AI_POLICY.md`. Your user needs to abide by this policy. On GitHub you MUST:

- **Mark all AI-generated content.** Text you put into an issue, PR, or comment goes under a
  section heading labeling it as AI-generated, or in a code/quote block.
- **Never emit only raw AI text as a reply.** AI content must carry human commentary
  explaining its relevance.
- **Do not submit code the user hasn't read.** Keep changes minimal, strip AI artifacts and
  needless complexity. Open PRs that are not ready or not reviewed by the user as drafts.
- **Check for duplicates.** Search open issues (and `gh search prs` for PRs) first; show a
  likely duplicate to the user before continuing. Skip if there is no internet access.

## Repository overview

DeepGuard is a deepfake detection project built around two hybrid CNN-ViT models
(**MS-EffViT**, **MS-EffGCViT**). Three parts with separate lifecycles:

- **`deepguard/`**, **`inference/`**, **`explainability/`**, **`preprocess/`** — the PyPI package.
- **`App/`** (FastAPI) + **`client/`** (React) — web app, excluded from the PyPI package.
- `train_eff_vit.py` / `train_eff_gcvit.py` — standalone training entrypoints, not packaged.

## Commands

```bash
pip install -r requirements.txt      # full dev environment (backend + ML + lint)
ruff check .                         # lint (default rules, no ruff config)
ruff format --check .
pytest --tb=short -q
```

There is no PR CI workflow for lint/tests, so run these locally before opening a PR.

### Training / inference

```bash
python -m train_eff_vit --root-dir DATA_ROOT --model-ver ms_eff_vit_b5 --dataset ff++ --seed 2025 --wandb-api-key <key>
python -m train_eff_gcvit --root-dir DATA_ROOT --model-ver ms_eff_gcvit_b0 --dataset celeb_df_v2 --seed 2025 --wandb-api-key <key>
python -m inference.predict_video --root-dir DATA_ROOT --model-name ms_eff_gcvit_b0 --model-dataset kodf --num-frames 20 --agg-mode conf
```

Per-model/per-dataset hyperparameters live in `deepguard/config/<model_ver>/<dataset>.yaml`
(OmegaConf); training is tracked with Weights & Biases.

### Backend (App/)

Run from inside `App/` — routes import siblings directly (`from routes import ...`,
`from services import ...`), not as a package:

```bash
cd App
uvicorn main:app --reload                     # port 8000
celery -A celery_app worker --loglevel=info   # async inference/explain tasks
```

Requires `App/.env` (`DB_*` for MySQL, `UPLOAD_DIR`/`EXPLAIN_UPLOAD_DIR`, `HF_TOKEN`,
`REDIS_HOST`/`REDIS_PORT`, `CORS_ALLOWED_ORIGINS`) plus running MySQL and Redis.
DB schema/seed SQL lives in `App/sql/`.

### Frontend (client/)

```bash
cd client
npm install
npm start      # :3000, proxies /api and /static/{uploads,explain} to 127.0.0.1:8000
npm test
npm run build
```

## Architecture

### Models (`deepguard/`)

Both models are registered with `timm` and share a dual-branch design:

1. **`FeatExtractor`** (`deepguard/layers/featextractor.py`) taps two EfficientNet feature maps:
   a **low-level / high-resolution** one (`l_block_idx` ∈ {0,1,2}) for local forgery artifacts
   (texture, blending, compression), and a **high-level / low-resolution** one
   (`h_block_idx` ∈ {4,6}) for global semantics (lighting, geometry, shadows).
2. Each map feeds its own transformer branch — window attention for MS-EffViT, GCViT-style
   local+global window attention for MS-EffGCViT — configured via `l_*`/`h_*` args.
3. Both branches combine into one binary logit (`num_classes=1`, real=0/fake=1).

Pretrained weights are fetched per `(model_name, dataset)` from GitHub Releases URLs in each
model file's `weight_registry`; mirrors are on Hugging Face Hub (`KoreaPeter/ms-eff-gcvit-deepfake`).

This low/high branch split is the central concept — it drives the XAI design, augmentation
choices, and CLI flags across training/inference.

### Explainability (`explainability/`)

CAM methods are assigned per branch based on empirical performance, not interchangeable:
- **low-level branch**: `HiResCAM`, `GradCAMElementWise`, `LayerCAM`
- **high-level branch**: `EigenGradCAM`, `GradCAM++`, `XGradCAM`

Only these six are supported; don't add others back without a reason.

- `BaseExplainer` (`explainer/base_explainer.py`): `branch_level` (`"low"`/`"high"`/`"both"`,
  default `"high"`) picks the branch. Target layers are fixed per branch: GCViT →
  `stages[-1].blocks[-1].norm2`, ViT → `blocks[-1].norm1`, `eff_b0`/`eff_b5` → last backbone
  feature map. `dataset` is optional (unused by DeepClip).
- `explainer/clip_explainer.py`: `Clip<Method>Explainer` for DeepClip (`deepclip_b0`/`deepclip_b5`,
  branches under `model.image_encoder`). deepclip is an optional extra — the guarded import in
  `explainability/utils/deepclip.py` raises at construction time, so `import explainability`
  works without it. Clip explainers target only the fake/real output; no per-artifact targets.

### Preprocessing (`preprocess/`)

Each dataset (`celeb_df_v2/`, `ff++/`, `kodf/`) has its own scripts because raw layouts differ.

1. **Face detection on real videos only** (`yolov8n-face.pt`, downloaded from the `v0.1.0`
   GitHub Release) — fake videos reuse their source real video's boxes (~80% less work).
2. **Face cropping + 5-point landmarks** (`.npy`) for both real and fake videos, used by
   landmark-based augmentation (`deepguard/data/dataset_cutout.py`).

Output layout `DATA_ROOT/{crops,landmarks}/{video_id}/*.{png,npy}` plus
`train_frame_metadata.csv` is what every `--root-dir` expects.

### Data pipeline (`deepguard/data/`)

Augmentation subclasses (`dataset_cutmix.py`, `dataset_cutout.py`, `dataset_mixup.py`) are
selected via YAML config (`mixup_alpha`/`mixup_prob`, `cutout_prob`). `handle_imbalance.py`
handles class imbalance (e.g. Celeb-DF-v2 is ~86% fake).

### Web app (`App/` + `client/`)

- Layering: `routes/` (thin, mounted under `/api`) → `services/` (logic, DB, file I/O; some are
  Celery tasks) → `db/`. `schemas/` holds Pydantic models.
- `App/db/database.py` has two async engines: pooled `engine` for requests, unpooled
  `celery_engine`/`celery_db_conn` for Celery tasks (outside the request lifecycle).
- Verdicts (`/inference`) use only v1 (MS-EffGCViT, `inference_svc.MODEL_CONFIG['v1']`).
  DeepClip is used only for the on-demand "20 artifact texts" feature (4 regions × 5 types,
  `PAIR_ORDER`): `routes/artifact.py` + `services/artifact_svc.py` (fast→`deepclip_b0`,
  pro→`deepclip_b5`), stored in `image_artifact_result` / `video_artifact_result` (video stores
  the mean over frames). Offered only for logged-in users' FAKE results, shown via
  `client/src/components/ArtifactTextPanel.js` (BETA).
- Session auth is cookie-based via Redis (`RedisSessionMiddleware`), not JWT.
- Uploads, videos, and CAM overlays are served from `/static`.

## Conventions

- Comments and docstrings in English. User-facing strings (UI text, API error messages,
  Pydantic `Field` descriptions, FastAPI `summary=`) and data values (e.g. `"서양인"`/`"동양인"`,
  KoDF CSV column names) stay in Korean.
- Keep `l_`/`h_` prefixes (low/high-level branch) when adding dual-branch parameters.
- `dataset` ∈ `celeb_df_v2`, `ff++`, `kodf`; `model_name` ∈ `ms_eff_vit_b0`, `ms_eff_vit_b5`,
  `ms_eff_gcvit_b0`, `ms_eff_gcvit_b5`.
