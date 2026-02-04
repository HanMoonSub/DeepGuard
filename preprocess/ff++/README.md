# FaceForensic++

Learning to Detect Manipulated Facial Images[[Paper]](https://arxiv.org/abs/1901.08971) [[Download]](https://github.com/ondyari/FaceForensics)

But, If you want to download quickly, there are FaceForensics++ in Kaggle Dataset [Click Here](https://www.kaggle.com/datasets/xdxd003/ff-c23)

## Overview

`FaceForensics++` is a forensics dataset consisting of 1000 original video sequences that have been manipulated with four automated face manipulation methods: `Deepfakes`, `Face2Face`, `FaceSwap` and `NeuralTextures`. The data has been sourced from 977 youtube videos and all videos contain a trackable mostly frontal face without occlusion with enables aumated tampering methods to generate realistic forgeries. As we provide binary masks the data can be used for image and video classification as well as segmentation. In addition, we provide 1000 Deepfakes models to generate and augment new data. 

## Kaggle Dataset Overview

This is the FaceForensics++ dataset downloaded from original scripts. The dataset contains the following folders: `DeepFake Detection`, `Deepfakes`, `Face2Face`, `FaceShifter`, `NeuralTextures`, `Original`, `CSV Files`. Total 7010 files with 7000 mp4 videos (6000 deepfake, 1000 real) and 10 csv files 

But, I'm gonna use only 5 deepfake model(`Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, `NeuralTextures`) not `DeepFake Detection`

| Source | Real/Fake | Videos | Description |
| ------ | ------ | ------- | ----------- |
| **`Deepfakes`** | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | 1000 Videos | Deep learning-based face replacement using autoencoders | 
| **`Face2Face`** | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | 1000 Videos | Facial reenactment that transfers expressions from source to target | 
| **`FaceSwap`** | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) |  1000 Videos | Graphics-based face replacement using traditional algorithms |
| **`FaceShifter`** | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) |  1000 Videos | High-fidelity face swapping with robust occlusion handling |
| **`NeuralTextures`** | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) |  1000 Videos | Facial reenactment using learned neural textures to modify expressions |
| **`Original`** | ![Real](https://img.shields.io/badge/Real-blue?style=flat-square) |  1000 Videos | Unaltered, authentic videos collected from YouTube |

## Split Data 

> _`There's no fixed test dataset. So we did split train and test data by 8:2`_

### Train Video Data

| DeepFake | Face2Face | FaceSwap | FaceShifter | NeuralTextures |  Original  |
| ---------- | ------------ | --------------- | ------- | -----------| ------- |
| ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Real](https://img.shields.io/badge/Real-blue?style=flat-square)|
|  796 | 798 | 793 | 810 | 807 | 796 | 


### Test Video Data

| DeepFake | Face2Face | FaceSwap | FaceShifter | NeuralTextures |  Original  |
| ---------- | ------------ | --------------- | ------- | -----------| ------- |
| ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Fake](https://img.shields.io/badge/Fake-red?style=flat-square) | ![Real](https://img.shields.io/badge/Real-blue?style=flat-square)|
|  204 | 202 | 207 | 190 | 193 | 204 | 

## How to Use Python Script

### 1. Split Data
This script integrates FaceForensics++ (FF++) metadata and splits the dataset into Train and Test sets based on the JSON split files(`train.json`, `test.json`).

> **ID Mapping**: Assigns unique ID prefixes to each manipulation method (e.g., 0_ for Deepfakes, 1_ for Face2Face) for easier video identification.

> **Metadata Integration**: Automatically links manipulated videos with their original counterparts to match frame counts and source information.


```python
# Ensure DATA_ROOT points to the directory containing the Kaggle dataset

python -m preprocess.ff++.split_data --root-dir DATA_ROOT --print-info True
```

> **Output**: train_metadata.csv, test_metadata.csv

| Column | Description |
| ------ | ----------- |
| `label` |  Ground truth label ('FAKE' or 'TRUE')          |
| `frame_cnt` |   Total number of frames in the video       |
| `d_vid` |  Duplicated Video name for each deepfake model        |
| `source` |  Deepfake model name(e.g., FaceSwap, Face2Face)        |
| `ori_vid` |  Video name used for manipulation        |
| `ori_framecnt` |   Total number of frames in the original video       |
| `vid` |   Unique Video name       |

### 2. Detect Original Face

To maximize preprocessing efficiency, face detection is performed only on original (real) videos. Since manipulated videos in FF++ share the same spatial coordinates as their sources, these bounding boxes are reused for the corresponding deepfake versions.

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
python -m preprocess.ff++.detect_original_face \
	--root-dir DATA_ROOT \
	--num-frames 20 \
	--conf-thres 0.5 \
	--min-face-ratio 0.01 \
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--root-dir` | (Required)  | Root directory of the FF++ dataset |
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
python -m preprocess.ff++.crop_face \
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
| ![](https://img.shields.io/badge/REAL-blue?style=flat-square) | Original Videos | 15,847 |
| ![](https://img.shields.io/badge/FAKEL-red?style=flat-square) | Deepfakes | 12,422 |
| ![](https://img.shields.io/badge/FAKEL-red?style=flat-square) | Face2Face | 12,685 |
| ![](https://img.shields.io/badge/FAKEL-red?style=flat-square) | FaceShifter | 12,847 |
| ![](https://img.shields.io/badge/FAKEL-red?style=flat-square) | FaceSwap | 12,416 |
| ![](https://img.shields.io/badge/FAKEL-red?style=flat-square) | NeuralTextures | 12,808 |
| Total |  | 79,025 |


## Citation

```
@inproceedings{roessler2019faceforensicspp,
	author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
	title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
	booktitle= {International Conference on Computer Vision (ICCV)},
	year = {2019}
}
```