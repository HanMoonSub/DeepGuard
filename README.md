# Deepfakes Detection

<p align="center">
  <img src="https://img.shields.io/github/license/HanMoonSub/DeepGuard?style=flat-square&color=555555&logo=github&logoColor=white" alt="License">
  <img src="https://img.shields.io/github/stars/HanMoonSub/DeepGuard?style=flat-square&color=FFD700&logo=github&logoColor=white" alt="Stars">
  <img src="https://img.shields.io/github/downloads/HanMoonSub/DeepGuard/total?style=flat-square&color=brightgreen&logo=github&logoColor=white" alt="Downloads">
  <img src="https://img.shields.io/github/last-commit/HanMoonSub/DeepGuard?style=flat-square&color=lightgrey&logo=github&logoColor=white" alt="Last Commit">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/Release-v0.2.0-orange?style=flat-square&logo=github&logoColor=white" alt="Release">
  <img src="https://img.shields.io/github/repo-size/HanMoonSub/DeepGuard?style=flat-square&color=blueviolet" alt="Repo Size">
</p>

<p align="center">
  <img src="docs/samples/deepfake_thumbnails.png" alt="DeepGuard Banner" width="800" height="400">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Deepfake_Detection-red?style=flat-square" alt="Task">
  <img src="https://img.shields.io/badge/Dataset-FaceForensics%2B%2B-blue?style=flat-square" alt="FF++">
  <img src="https://img.shields.io/badge/Dataset-Celeb--DF(v2)-green?style=flat-square" alt="Celeb-DF">
  <img src="https://img.shields.io/badge/Dataset-KODF-yellowgreen?style=flat-square" alt="KODF">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-MS--Eff--GCViT--B0%20%2F%20B5-orange?style=flat-square" alt="Models">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/W%26B-Recording-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=white" alt="W&B">
</p>

<p align="center">
  <a href="README_KR.md"><b>🇰🇷 한국어 버전</b></a> | 
  <a href="README_JP.md"><b>🇯🇵 日本語版</b></a> | 
  <a href="#-model-evaluation"><b>📈 Model Evaluation</b></a> | 
  <a href="#-predict-image--video"><b>🔮 Try Demo</b></a>
</p>

## 📌 Contents

- [💡 Install & Requirements](#-install--requirements)
- [🛠 SetUp](#-setup)
- [📚 DeepFake Video BenchMark Datasets](#-deepfake-video-benchmark-datasets) — Overview of Celeb-DF-v2, FF++, and KoDF datasets used for training.
- [⚙️ Data Preparation](#data-preparation) — Efficient face detection and landmark extraction pipeline using YOLOv8
- [🏗 Model Architecture](#-model-architecture) — Detailed look into our hybrid CNN-ViT (MS-EffViT & MS-EffGCViT) designs.
- [🧬 Model Zoo](#-model-zoo) — Comparison of model variants, parameter counts, and computational complexity (FLOPs).
- [🚀 Training](#-training) - Step-by-step training scrips with Goolge Colab and W&B experiment tracking
- [📈 Model Evaluation](#-model-evaluation) - Benchmarking results
- [💻 Model Usage](#-model-usage) - How to integrate DeepGuard models into your own Python code or via timm
- [🔮 Predict Image & Video](#-predict-image--video) - Simple Inference examples for detecting deepfakes in image and video
- [🎨 DeepFake AI Explainability](#-deepfake-ai-explainability) - Visualizing model focus using Grad-CAM and attention maps
- [📊 Metrics and Evaluation for DeepFake XAI](#-metrics-and-evaluation-for-deepfake-xai) - Quantitative assessment of explanation reliability
- [📬 Authors](#-authors)
- [📝 Reference](#-reference)
- [⚖️ License](#-license)

## 💡 Install & Requirements

To install requirements: 

```python
pip install -r requirements.txt
```

## 🛠 SetUp

Clone the repository and move into it:
```
git clone https://github.com/HanMoonSub/DeepGuard.git

cd DeepGuard
```

## 📚 DeepFake Video BenchMark Datasets

To evaluate the generalization and robustness of our deepfake detection model, we utilize three large-scale, widely recognized benchmark datasets. Each dataset presents unique challenges and covers different types of forgery methods.

| Dataset | Real Videos | Fake Videos | Year | Participants | Description (Paper Title) | Details |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Celeb-DF-v2** | 890 | 5,639 | 2019 | 59 | *A Large-scale Challenging Dataset for DeepFake Forensics* | [🔗 Readme](preprocess/celeb_df_v2/README.md) |
| **FaceForensics++** | 1,000 | 6,000 | 2019 | 1,000 | *Learning to Detect Manipulated Facial Images* | [🔗 Readme](preprocess/ff++/README.md) |
| **KoDF** | 62,166 | 175,776 | 2020 | 400 | *Large-Scale Korean Deepfake Detection Dataset* | [🔗 Readme](preprocess/kodf/README.md) |
 
<div id="data-preparation"></div>

## ⚙️ Data Preparation

Our preprocessing pipeline is designed to efficiently extract facial features from videos and prepare them for high-accuracy deepfake detection.

### Detect Original Face
To maximize preprocessing efficiency, <ins>**face detection is performed only on original (real) videos.**</ins> Since mnipulated videos in DeepFake Video BenchMark Datasets share the same spatial coordinates as their sources, these bounding boxes are reused for the corresponding deepfake versions.

🚀 Efficiency Optimizations
- **Lightweight Model**: Uses yolov8n-face for high-speed inference without sacrificing accuracy.

- **Targeted Processing**: By detecting faces only in original videos, the total detection workload is reduced by approximately 80%.

- **Dynamic Rescaling**: To maintain consistent inference speed across different resolutions, frames are automatically resized based on their dimensions:

| Frame Size(Longest Side) | Scale Factor | Action |
| ------------------------ | ------------ | ------ |
|         < 300px          |      2.0     |  ![](https://img.shields.io/badge/Upscale-green?style=flat-square) |
|      300px - 700px       |      1.0     | ![](https://img.shields.io/badge/No_Change-gray?style=flat-square) |
|      700px - 1500px      |      0.5     | ![](https://img.shields.io/badge/DownScale-skyblue?style=flat-square) |
|          > 1500px        |      0.33    | ![](https://img.shields.io/badge/DownScale-skyblue?style=flat-square) | 

### Face Cropping & Landmark Extraction
This module extracts face crops from both original and deepfake videos using the bounding boxes generated in the previous step. It also performs landmark detection to facilitate advanced augmentations like Landmark-based Cutout

🛠 Key Features
- **Dynamic Margin with Jitter**: Adds a configurable margin around the face. The margin_jitter parameter introduces random variance to the crop size, making the model more robust to different face scales.

- **Landmark Localization**: `Detects 5 primary facial landmarks` (eyes, nose, mouth corners) and saves them as .npy files.

```Plaintext
DATA_ROOT/
├── crops/
│   └── {video_id}/
│       ├── 12.png
│       └── ...
├── landmarks/
│   └── {video_id}/
│       ├── 12.npy
│       └── ...
└── train_frame_metadata.csv
```

### Dataset-Specific Pipelines

Click the links below to view the specific preprocessing details for each dataset:

* [**Celeb-DF V2 Preprocess**](/preprocess/celeb_df_v2/README.md)
* [**FaceForensics++ Preprocess**](/preprocess/ff++/README.md)
* [**KoDF Preprocess**](/preprocess/kodf/README.md)

## 🏗 Model Architecture

Multi Scale Efficient Global Context Vision Transformer is an optimized multi-scale hybrid architecture that integrates CNN-driven spatial inductive bias with hierarchical attention mechanisms to effectively identify subtle(local) artifacts and macro(global) artifacts for robust deepfake forensics."

#### Explore More Details

- [Model Architecture: MS-EffViT](deepguard/MS_EffViT.md) - _**Multi Scale Efficient Vision Transformer**_

- [Advanced Architecture: MS-EFFGCViT](deepguard/MS_EffGCViT.md) - _**Multi Scale Efficient Global Context Vision Transformer**_


<p align="center">
  <img src="docs/architectures/ms_eff_gcvit.JPG" width="100%" height="700">
</p>

We utilizes two distinct types of self-attention to capture both long-range and short-range information across feature maps.

- **Local Window Attention**: this model efficiently captures local textures and precise spatial details while maintaining linear computational complexity relative to the image size.

- **Global Window Attention**: Unlike Swin Transformer, this module utilizes global-queries that interact with local window keys and values. This allows each local region to incorporate global context, effectively capturing long-range dependencies and providing a comprehensive understanding of the entire spatial structure


<p align="center">
  <img src=docs/architectures/window_attention.JPG width="100%" height="300">
</p>

## 🧬 Model Zoo

| Model | Resolution | # Total Params(M) | # Backbone(M) | # L-ViT(M) | # H-ViT(M)  | FLOPs (G) | Model Config |
| ----- | ---------- | -------------- | ----------- |------------- | ------------- | --------------  | ------- | 
| ⚡ ms_eff_gcvit_b0 | 224 X 224 | 8.7 | 3.6(41.4%) | 1.7(19.5%) | 3.3(37.9%) | 0.87 |  [spec](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |
| 🔥 ms_eff_gcvit_b5 | 384 X 384 | 50.3 | 27.3(54.3%) | 6.6(13.1%) | 16.1(32.0%) | 13.64 | [spec](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

## 🚀 Training

We provide training scripts for both `ms_eff_vit` and `ms_eff_gcvit`. We recommend using **Google Colab** for free GPU access and **Weightes & Biases(W&B)** for experiment tracking

#### 📊 Weight & Biases Experiments

* **ms_eff_vit_b0:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_vit_b0_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_vit_b0_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_vit_b0_kodf)
* **ms_eff_vit_b5:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_vit_b5_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_vit_b5_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_vit_b5_kodf)
* **ms_eff_gcvit_b0:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b0_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b0_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b0_kodf)
* **ms_eff_gcvit_b5:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b5_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b5_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b5_kodf)

```python
!python -m train_eff_vit \ # train_eff_gcvit
    --root-dir DATA_ROOT \ 
    --model-ver "ms_eff_vit_b5" \ # ms_eff_vit_b0, ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5
    --dataset "ff++" \ # ff++, celeb_df_v2, kodf
    --seed 2025 \ # for reproducibility
    --wandb-api-key "your-api-key" # Write your own api key
```

## 📈 Model Evaluation

```python
!python -m inference.predict_video \
    --root-dir DATA_ROOT \
    --margin-ratio 0.2 \
    --conf-thres 0.5 \
    --min-face-ratio 0.01 \
    --model-name "ms_eff_gcvit_b0" \ # ms_eff_vit_b0, ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5
    --model-dataset "kodf" \ # ff++, celeb_df_v2, kodf
    --num-frames 20 \
    --tta-hflip 0.0 \
    --agg-mode "conf" \
```


**Celeb DF(v2) Pretrained Models**

| Model Variant | Test@Acc | Test@Auc | Test@log_loss | Download | Train Config |
| ------------- | -------- | -------- | ---------- | -------- | ----- |
| ms_eff_gcvit_b0 | 0.9842 | 0.9965 | 0.0283 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b0_celeb_df_v2.bin) | [recipe](deepguard/config/ms_eff_gcvit_b0/celeb_df_v2.yaml) |
| ms_eff_gcvit_b5 | 0.9981 | 0.9984 | 0.0089 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b5_celeb_df_v2.bin) | [recipe](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

**FaceForensics++ Pretrained Models**

| Model Variant | Test@Acc | Test@Auc | Test@log_loss | Download | Train Config |
| ------------- | -------- | -------- | ---------- | -------- | ------ |
| ms_eff_gcvit_b0 | 0.9808 | 0.9969 | 0.0637| [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b0_ff++.bin) | [recipe](deepguard/config/ms_eff_gcvit_b0/celeb_df_v2.yaml) |
| ms_eff_gcvit_b5 | 0.9850 | 0.9974 | 0.0492 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b5_ff++.bin) | [recipe](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

**KoDF Pretrained Models**

| Model Variant | Test@Acc | Test@Auc | Test@log_loss | Download | Train Config |
| ------------- | -------- | -------- | ---------- | -------- | ------ |
| ms_eff_gcvit_b0 | 0.9655 | 0.9792 | 0.1237 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.2.0/ms_eff_gcvit_b0_kodf.bin) | [recipe](deepguard/config/ms_eff_gcvit_b0/celeb_df_v2.yaml) |
| ms_eff_gcvit_b5 | 0.9850 | 0.9974 | 0.0492 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.2.0/ms_eff_gcvit_b5_kodf.bin) | [recipe](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

## 💻 Model Usage

**Quick Start**
You can load the models directly via the `DeepGuard` package or through the `timm` interface.

**Available Datasets**: `celeb_df_v2`, `ff++`, `kodf`

**Installation**

```bash
# pip install -U git+https://github.com/HanMoonSub/DeepGuard.git
pip install deepguard
```


**Option A: Direct Import (via DeepGuard)**

```python
from deepguard import ms_eff_gcvit_b0, ms_eff_gcvit_b5

model = ms_eff_gcvit_b0(pretrained=True, dataset="celeb_df_v2")
model = ms_eff_gcvit_b5(pretrained=True, dataset="ff++")
```

**Option B: Using timm Interface (via timm)**

```python
import timm
import deepguard

model = timm.create_model("ms_eff_gcvit_b0", pretrained=True, dataset="ff++")
model = timm.create_model("ms_eff_gcvit_b5", pretrained=True, dataset="kodf")
```

**Option C: Hugging Face Hub**

```python
import torch
from huggingface_hub import hf_hub_download
from deepguard import ms_eff_gcvit_b0  # or ms_eff_gcvit_b5

REPO_ID = "KoreaPeter/ms-eff-gcvit-deepfake"

ckpt = hf_hub_download(REPO_ID, "ms_eff_gcvit_b0_kodf.bin")  # celeb_df_v2 | ff++ | kodf
model = ms_eff_gcvit_b0(pretrained=False)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
```

## 🔮 Predict Image & Video

#### Predict DeepFake Image

```python

from inference.image_predictor import ImagePredictor

# Initialize the predictor
predictor = ImagePredictor(
            margin_ratio = 0.2, #  Margin ratio around the detected face crop
            conf_thres = 0.5, # Confidence threshold for face detection
            min_face_ratio = 0.01, # Minimum face-toframe size ratio to process 
            model_name = "ms_eff_vit_b0", #  ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5  
            dataset = "celeb_df_v2" # ff++, kodf
            )

# Run Inference
result = predictor.predict_img(
            img_path="path/to/image.jpg",
            tta_hflip=0.0 # Horizontal Flip for Test-Time Augmentation 
            )

print(f"Deepfake Probability: {result:.4f}")

```
#### Predict DeepFake Video

```python

from inference.video_predictor import VideoPredictor

# Initialize the predictor
predictor = VideoPredictor(
            margin_ratio = 0.2, #  Margin ratio around the detected face crop
            conf_thres = 0.5, # Confidence threshold for face detection
            min_face_ratio = 0.01, # Minimum face-toframe size ratio to process 
            model_name = "ms_eff_vit_b0", #  ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5  
            dataset = "celeb_df_v2" # ff++, kodf
            )

# Run Inference
result = predictor.predict_video(
            video_path = "path/to/video.mp4",
            num_frames = 20, # Number of frames to sample per video
            agg_mode = "conf", # Aggregation Method: 'conf', 'mean', 'vote'
            tta_hflip=0.0 # Horizontal Flip for Test-Time Augmentation 
            )

print(f"Deepfake Probability: {result:.4f}")

```

## 🎨 DeepFake AI Explainability

Deepfake detection is only as trustworthy as its explanations. DeepGuard integrates a production-ready XAI Toolkit that visualizes where and why the model flags a face as manipulated — turning a black-box score into actionable forensic evidence.

⭐ Validated on hybrid CNN-ViT architectures, specifically `MS-EffViT` and `MS-EffGCViT`.  
⭐ Dual-Branch Analysis: Dual-branch design mirrors the model's own multi-scale reasoning

### 🧠 How Dual-Branch XAI Works

<p align="center">
  <img src="docs/architectures/dual_branch.gif" width="70%" height="300">
</p>


| Branch | Feature Map | Focus | Best For |
| ------ | ----------- | ----- | -------- |
| ![](https://img.shields.io/badge/Low_level-blue?style=flat-square)       | High Resolution | Local Forgery artifacts | Skin texture, boundary blending, compression traces |
| ![](https://img.shields.io/badge/High_level-red?style=flat-square)       | Low Resolution  | Global Semantic Structure | Lighting inconsistency, facial geometry, Shadow artifacts | 

### 📐 XAI Methods

Each method is assigned to the branch where it performs best empirically.

| Branch | Method | 🎯 Core Idea |
| :--- | :--- | :--- |
| **`low level`** | **HiResCAM** | Like GradCAM but element-wise multiply the activations with the gradients; provably guaranteed faithfulness for certain models |
| **`low level`** | **GradCAMElementWise** | Like GradCAM but element-wise multiply the activations with the gradients then apply a ReLU operation before summing |
| **`low level`** | **LayerCAM** | Spatially weight the activations by positive gradients. Works better especially in lower layers |
| --- | --- | --- | --- |
| **`high level`** | **EigenGradCAM** | Like EigenCAM but with class discrimination: First principle component of Activations*Grad. Looks like GradCAM, but cleaner |
| **`high level`** | **GradCAM++** |  Like GradCAM but uses second order gradients |
| **`high level`** | **XGradCAM** | Like GradCAM but scale the gradients by the normalized activations |

- **`aug_smooth`** applies TTA (horizontal flips) before averaging CAMs → smoother, more object-aligned maps  
- **`eigen_smooth`** applies PCA noise reduction → retains dominant forgery pattern only

### 💡 DeepFake XAI Usage

**Low-Level Branch — Local Artifact Detection**

```python
from explainability import HiResCAMExplainer, GradCAMElementWiseExplainer, LayerCAMExplainer

explainer = HiResCAMExplainer(
    model_name   = "ms_eff_gcvit_b0",  # or ms_eff_vit_b0, ms_eff_gcvit_b5, ms_eff_vit_b5
    dataset      = "celeb_df_v2",       # or ff++, kodf
    branch_level = "low",
)
```

**High-Level Branch — Global Semantic Detection**
```python
from explainability import EigenGradCAMExplainer, GradCAMPlusPlusExplainer, XGradCAMExplainer

explainer = EigenGradCAMExplainer(
    model_name   = "ms_eff_gcvit_b0",
    dataset      = "celeb_df_v2",
    branch_level = "high",
)
```

### 🎨 Visualization Modes

<p align="center">
  <img src="docs/xai-results/xai_demo.gif" width="80%" height=300>
</p>

**1. Heatmap — Continuous activation distribution**

```python
result = explainer.display_heatmap_on_image(
    img_path     = "path/to/image.jpg",
    category     = 1,      # 0: Real, 1: Fake
    threshold    = 0.5,    # binarization cutoff (0.5~1.0), or "auto" for Otsu
    image_weight = 0.5,    # 0.0: heatmap only ← → 1.0: original only
    aug_smooth   = False,  # TTA smoothing (not supported on 'pro' models)
    eigen_smooth = False,  # PCA noise reduction
)
```

**2. Bounding Box — Discrete forgery region localization**

```python
result = explainer.display_bbox_on_image(
    img_path     = "path/to/image.jpg",
    category     = 1,
    threshold    = 0.5,
    thickness    = 1,
    aug_smooth   = False,
    eigen_smooth = False,
)
```

**3. Heatmap + BBox — Full overlay (recommended for reporting)**
```python
result = explainer.display_heatmap_bbox_on_image(
    img_path     = "path/to/image.jpg",
    category     = 1,
    threshold    = 0.5,
    image_weight = 0.5,
    aug_smooth   = False,
    eigen_smooth = False,
)
```

### 📊 Visual Results

<p align="center">
  <table>
    <tr>
      <td><img src="docs/architectures/low_branch.gif" width="100%"></td>
      <td width="20%"></td>
      <td><img src="docs/architectures/high_branch.gif" width="100%"></td>
    </tr>
  </table>
</p>

#### MS-EFF-VIT — Low-Level Branch

| Model | Branch-Level | Image | HiresCam | GradCamElementwise | LayerCam |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **⚡ ms-eff-vit-b0** | ![](https://img.shields.io/badge/Low_level_Branch-blue?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b0_low_hirescam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b0_low_gradcamelementwise.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b0_low_layercam.JPG" width="100"> |
| **🔥 ms-eff-vit-b5** | ![](https://img.shields.io/badge/Low_level_Branch-blue?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b5_low_hirescam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b5_low_gradcamelementwise.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b5_low_layercam.JPG" width="100"> |

#### MS-Eff-ViT — High-Level Branch

| Model | Branch-Level | Image | EigenGradCam | GradCamPlusPlus | XGradCam |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **⚡ ms-eff-vit-b0** | ![](https://img.shields.io/badge/High_level_Branch-red?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b0_high_eigengradcam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b0_high_gradcamplusplus.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b0_high_xgradcam.JPG" width="100"> |
| **🔥 ms-eff-vit-b5** | ![](https://img.shields.io/badge/High_level_Branch-red?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b5_high_eigengradcam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b5_high_gradcamplusplus.JPG" width="100"> | <img src="docs/xai-results/ms_eff_vit_b5_high_xgradcam.JPG" width="100"> |

#### MS-EFF-GCVIT — Low-Level Branch

| Model | Branch-Level | Image | HiresCam | GradCamElementwise | LayerCam |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **⚡ ms-eff-gcvit-b0** | ![](https://img.shields.io/badge/Low_level_Branch-blue?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b0_low_hirescam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b0_low_gradcamelementwise.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b0_low_layercam.JPG" width="100"> |
| **🔥 ms-eff-gcvit-b5** | ![](https://img.shields.io/badge/Low_level_Branch-blue?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b5_low_hirescam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b5_low_gradcamelementwise.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b5_low_layercam.JPG" width="100"> |

#### MS-Eff-GCViT — High-Level Branch

| Model | Branch-Level | Image | EigenGradCam | GradCamPlusPlus | XGradCam |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **⚡ ms-eff-gcvit-b0** | ![](https://img.shields.io/badge/High_level_Branch-red?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b0_high_eigengradcam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b0_high_gradcamplusplus.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b0_high_xgradcam.JPG" width="100"> |
| **🔥 ms-eff-gcvit-b5** | ![](https://img.shields.io/badge/High_level_Branch-red?style=flat-square) | <img src="docs/samples/images/western/western_fake_1.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b5_high_eigengradcam.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b5_high_gradcamplusplus.JPG" width="100"> | <img src="docs/xai-results/ms_eff_gcvit_b5_high_xgradcam.JPG" width="100"> |


## 📊 Metrics and Evaluation for DeepFake XAI 


## 📬 Authors

_**This project was developed as a Senior Graduation Project by the Department of Software at Chungbuk National University (CBNU), Republic of Korea.**_

* **한문섭**: **Data & Backend Engineering** (Data Preprocessing Pipeline, DB Schema Design) — [hanmoon3054@gmail.com](mailto:hanmoon3054@gmail.com)
* **이예솔**: **UI/UX & Frontend Engineering** (UI/UX Design, User Dashboard, Model Visualization) — [yesol4138@chungbuk.ac.kr](mailto:yesol4138@chungbuk.ac.kr)
* **서윤제**: **AI Engineering** (AI Model Architecture, Inference API Design, Model Serving) — [seoyunje2001@gmail.com](mailto:seoyunje2001@gmail.com)


## 📝 Reference

1. [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) - _Pretrained Face Detection(MTCNN) and Recognition(InceptionResNet) Models by Tim Esler_
2. [`face-cutout`](https://github.com/sowmen/face-cutout) - _Face Cutout Library by Sowmen_
3. [`Celeb-DF++`](https://github.com/OUC-VAS/Celeb-DF-PP) - _Celeb-DF++ Dataset by OUC-VAS Group_
4. [`DeeperForensics-1.0`](https://github.com/EndlessSora/DeeperForensics-1.0) - _DeeperForensics-1.0 Dataset by Endless Sora_
5. [`Deepfake Detection`](https://github.com/abhijithjadhav/Deepfake_detection_using_deep_learning) - _Detection of Video Deepfake using ResNext and LSTM by Abhijith Jadhav_
6. [`deepfake-detection-project-v4`](https://github.com/ameencaslam/deepfake-detection-project-v4) - _Multiple Deep Learning Models by Ameen Caslam_
7. [`Awesome-Deepfake-Detection`](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection
) - _A curated list of tools, papers and code by Daisy Zhang_
8. [`Pytorch-Grad-Cam`](https://github.com/jacobgil/pytorch-grad-cam) - _Advanced Visual Explanations for PyTorch Models_

## ⚖️ License 

This project is licensed under the terms of the MIT license.
