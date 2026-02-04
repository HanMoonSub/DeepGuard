# Celeb_DF (V2)

A Large-scale Challenging Dataset for DeepFake Forensics [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf) [[Download]](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master)

But, If you want to download quickly, there are Celeb_DF (V2) in Kaggle Dataset [Click Here](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)

## Overview

Celeb-DF dataset include 590 original videos collected from YouTube with subjects of different ages, ethic groups and genders, and 5639 corresponding DeepFake Videos.


| Source | Real/Fake | Videos | Description |
| ------ | ------ | ------- | ----------- |
| **`celeb-real`** | ![Real](https://img.shields.io/badge/Real-blue?style=flat-square) | 590 Videos | Celebrity videos downloaded from YouTube | 
| **`youtube-real`** | ![Real](https://img.shields.io/badge/Real-blue?style=flat-square) | 300 Videos | Additional videos downloaded from YouTube | 
| **`celeb-synthesis`** | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) |  5639 Videos | Synthesized videos from Celeb-real |
| **`List_of_testing_videos.txt`** | ![Real](https://img.shields.io/badge/Real-blue?style=flat-square) ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) |  518 Videos | Real and Fake Vidoes for Test| 

## Split Data 

### Train Video Data

| celeb-real | youtube-real | celeb-synthesis | `Real Ratio` | `Fake Ratio` |
| ---------- | ------------ | --------------- | ------- | -----------|
|    482     |     230      |    4284  |  14.3%  | 85.7% | 

### Test Video Data

| celeb-real | youtube-real | celeb-synthesis | `Real Ratio` | `Fake Ratio` |
| ---------- | ------------ | --------------- | ------- | -----------|
|    108     |     70      |    340  |  34.4%  | 65.6% | 



## How to Use Python Script

### 1. Split Data
This script integrates Celeb_DF(V2) metadata and splits the dataset into Train and Test sets based on the `List_of_testing_videos.txt`

```python
# Ensure DATA_ROOT points to the directory containing the Kaggle dataset

python -m preprocess.celeb_df_v2.split_data --root-dir DATA_ROOT --print-info True
```

> **Output**: train_metadata.csv, test_metadata.csv

| Column | Description |
| ------ | ----------- |
| `label` |  Ground truth label ('FAKE' or 'TRUE')          |
| `source` |  `youtube-real`, `celeb-synthesis`, `celeb-real`   |
| `ori_vid` |  Video name used for manipulation        |
| `vid` |   Unique Video name       |

### 2. Detect Original Face

To maximize preprocessing efficiency, face detection is performed only on original (real) videos. Since manipulated videos in Celeb_DF(V2) share the same spatial coordinates as their sources, these bounding boxes are reused for the corresponding deepfake versions.

🚀 Efficiency Optimizations
- **Lightweight Model**: Uses yolov8n-face for high-speed inference without sacrificing accuracy.

- **Targeted Processing**: By detecting faces only in original videos, the total detection workload is reduced by approximately 80%.

- **Dynamic Rescaling**: To maintain consistent inference speed across different resolutions, frames are automatically resized based on their dimensions:

| Frame Size(Longest Side) | Scale Factor | Action |
| ------------------------ | ------------ | ------ |
|         < 300px          |      2.0     |  ![](https://img.shields.io/badge/Upscale-green?style=flat-square) |
|      300px - 700px       |      1.0     | ![](https://img.shields.io/badge/No_Change-gray?style=flat-square) |
|      700px - 1500px      |      0.5     | ![](https://img.shields.io/badge/No_Change-yellow?style=flat-square) |
|          > 1500px        |      0.33    | ![](https://img.shields.io/badge/No_Change-yellow?style=flat-square) | 

```python
python -m preprocess.celeb_df_v2.detect_original_face \
	--root-dir DATA_ROOT \
	--num-frames 20 \
	--conf-thres 0.5 \
	--min-face-ratio 0.01 \
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--root-dir` | (Required)  | Root directory of the celeb_df_v2 dataset |
| `--num-frames` | 10 | Number of frames to sample from each video |
| `--conf-thres` | 0.5 | Confidence of threshold for the Face Detector |
| `--min-face-ratio` | 0.01 | Minimum area ratio a face must occupy to be saved |
| `--jitter` | 0 | Random offset range applied to frame indices for diversity

📂 **Output Structure**

The detected bounding boxes are saved as individual JSON files named after the original video ID.

```Plaintext

DATA_ROOT/
└── boxes/
    ├── 000.json
    ├── 001.json
    └── ...
```

### 3. Face Cropping & Landmark Extraction
This module extracts face crops from both original and deepfake videos using the bounding boxes generated in the previous step. It also performs landmark detection to facilitate advanced augmentations like Landmark-based Cutout.

🛠 Key Features
- **Dynamic Margin with Jitter**: Adds a configurable margin around the face. The margin_jitter parameter introduces random variance to the crop size, making the model more robust to different face scales.

- **Landmark Localization**: `Detects 5 primary facial landmarks` (eyes, nose, mouth corners) and saves them as .npy files.

- **Frame-level Metadata**: Generates a comprehensive train_frame_metadata.csv mapping every saved crop to its label, source, and original video ID.

```bash
python -m preprocess.celeb_df_v2.crop_face \
    --root-dir DATA_ROOT \
    --margin-ratio 0.2 \
    --margin-jitter 0.0
```
| Argument | Default | Description | 
| -------- | ------- | ----------- |
| `--margin-ratio` | 0.2 | Base padding ratio around the detected bounding box |
| `--margin-jitter` | 0.0 | Intenstiy of random noise added to the margin for each crop | 


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

## Result of Data Processing

_**`Sampling`**_: <ins/>`num_frames: 20` per video with a `jitter: 5` (random frame offset)</ins>

_**`Face Quality`**_: <ins/>`min_face_ratio: 0.01` (to exclude tiny/low-quality faces)</ins>

_**`Crop Settings`**_: <ins/>`margin_ratio: 0.2` with `margin_jitter: 0`</ins>

| Category | Source Method | Frame Count | 
| -------- | ------------- | ----------- |
| ![](https://img.shields.io/badge/REAL-blue?style=flat-square) | Celeb-real | 9603 |
| ![](https://img.shields.io/badge/REAL-blue?style=flat-square) | YouTube-real | 4572 |
| ![](https://img.shields.io/badge/FAKE-red?style=flat-square) | Celeb-syntehsis | 85,495 |


## Citation

```
@inproceedings{Celeb_DF_cvpr20,
   author = {Yuezun Li, Xin Yang, Pu Sun, Honggang Qi and Siwei Lyu},
   title = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
   booktitle= {IEEE Conference on Computer Vision and Patten Recognition (CVPR)},
   year = {2020}
}
```
