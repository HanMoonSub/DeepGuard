# Deepfakes Detection

<p align="center">
  <img src="https://raw.githubusercontent.com/HanMoonSub/DeepGuard/main/Images/deepfake2.png" alt="DeepGuard Banner" width="800" height="400">
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/HanMoonSub/DeepGuard?style=flat-square&color=555555&logo=github&logoColor=white" alt="License">
  <img src="https://img.shields.io/github/stars/HanMoonSub/DeepGuard?style=flat-square&color=FFD700&logo=github&logoColor=white" alt="Stars">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" alt="Status">
  <img src="https://img.shields.io/badge/Release-v0.2.0-orange?style=flat-square&logo=github&logoColor=white" alt="Release">
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
  <img src="https://raw.githubusercontent.com/HanMoonSub/DeepGuard/main/Images/ms_eff_gcvit.JPG" width="100%" height="700">
</p>

We utilizes two distinct types of self-attention to capture both long-range and short-range information across feature maps.

- **Local Window Attention**: this model efficiently captures local textures and precise spatial details while maintaining linear computational complexity relative to the image size.

- **Global Window Attention**: Unlike Swin Transformer, this module utilizes global-queries that interact with local window keys and values. This allows each local region to incorporate global context, effectively capturing long-range dependencies and providing a comprehensive understanding of the entire spatial structure


<p align="center">
  <img src=https://raw.githubusercontent.com/HanMoonSub/DeepGuard/main/Images/window_attention.JPG width="100%" height="300">
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
pip install -U git+https://github.com/HanMoonSub/DeepGuard.git
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

## ⚖️ License 

This project is licensed under the terms of the MIT license.
