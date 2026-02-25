# KoDF: Large-Scale Korean Deepfake Detection Dataset

The KoDF dataset is a comprehensive collection of synthesized videos provided by AI-Hub, specifically designed to advance the detection of neural-network-based facial manipulations.

## Background: AI-Hub & NIA
AI-Hub is a national AI infrastructure platform operated by the National Information Society Agency (NIA) of South Korea. It serves as a cornerstone for South Korean AI development, providing high-quality datasets, software APIs, and computing resources to the public to foster innovation in the AI sector.

## Technical Specifications & Statistics
The dataset is characterized by its high volume and the diversity of its subjects, ensuring that detection models trained on this data are robust against various environmental and physiological factors.



## Data Statics

- [x] **Number of Subjects**: 400 participants
- [x] **Videos per Subject**: 150+ videos
- [x] **Total Data Duration**: 88.5 days
- [x] **Deepfake Model Variants**: 6 types

| Metric | Original Data | Fake Data |
| ------ | ------------- | --------- | 
| Total Videos | 62,166 | 175,776 |
| Average Video Length | 90+ second | 15+ second | 
| Total Duration | 1,500+ hours | 625+ hours|
| Resolution | 1920 X 1080 | 1920 X 1080 |
| FPS | 30 FPS | 30 FPS |
| Total Frames | 162,000,000+ | -- |

|  Column | Description | Example |
| ------- | ----------- | ------- |
| Video ID |  Unique identifer for the video  | 200715_23d3dsdfsdf34_1 | 
| User UUID | Unique identifier for the recorder | 23d3dsdfsdf001 |
| Start Date | Recording start timestamp | 2020.07.24:15.05.30 |
| Emotion Class | neutral, negative, positive | neutral |
| Label/Authenticity | Original(Real) vs Synthetic(Fake) | Real |
| Glasses | Presence of eyeglasses | No | 
| Idle State | Static frontal view without speech | True |
| Location | Indoor / Outdoor | Indoor |
| Light Source | Natural Light / Artificial light | Natural Light |
| Illumination Level | Bright, Medium, Low | Bright |
| Noise Level | Quiet, Moderate, Noisy | Quiet |

```mermaid
graph TD
    %% Main Root
    Root[Original Video] --> A[Crowdsourcing]
    Root --> B[Studio Recording]

    %% Crowdsourcing Branch
    A --> A1[Script-based]
    A --> A2[Scenario-based]
    A --> A3[IDLE]

    %% Crowdsourcing - Script Statistics
    A1 --> A1_1["<b>Neutral</b><br/>12,950 (21.6%)"]
    A1 --> A1_2["<b>Positive</b><br/>6,650 (11.1%)"]
    A1 --> A1_3["<b>Negative</b><br/>6,650 (11.1%)"]

    %% Crowdsourcing - Scenario Statistics
    A2 --> A2_1["<b>Neutral</b><br/>12,950 (21.6%)"]
    A2 --> A2_2["<b>Positive</b><br/>6,650 (11.1%)"]
    A2 --> A2_3["<b>Negative</b><br/>6,300 (10.5%)"]

    %% Crowdsourcing - IDLE
    A3 --- A3_1["350 (0.6%)"]

    %% Studio Branch
    B --> B1[Script-based]

    %% Studio - Script Statistics
    B1 --> B1_1["<b>Neutral</b><br/>3,750 (6.3%)"]
    B1 --> B1_2["<b>Positive</b><br/>1,900 (3.2%)"]
    B1 --> B1_3["<b>Negative</b><br/>1,850 (3.1%)"]

    %% Styling
    style Root fill:#f9f,stroke:#333,stroke-width:2px
    style A fill:#f2f2f2,stroke:#333
    style B fill:#f2f2f2,stroke:#333
    style A1 fill:#fff,stroke:#999
    style A2 fill:#fff,stroke:#999
    style B1 fill:#fff,stroke:#999
```

## How to Collect Data

### 1. Source Data Acquisition

The dataset utilizes **400 Korean participants** to ensure ethnic demographic accuracy.

**`Controlled Environment`**: <ins/>50 participants recorded in studio/home settings.</ins>

**`Crowdsourced`**: <ins/>350 participants recorded via mobile devices in diverse lighting/backgrounds.</ins>

**`Scripted Session`**: <ins/>Participants read specific passages to capture controlled lip movements.</ins>

**`Spontaneous Session`**: <ins/>Free-form Q&A to capture natural expressions and micro-movements.</ins>

---

### 2. Synthesis Techniques (6 Deepfake Models)

The dataset employs **six state-of-the-art (SOTA)** models categorized into two primary manipulation types:

#### ![](https://img.shields.io/badge/Face_Swap-7DF9FF?style=flat-square) 
**The process of replacing the identity of a "target" person with a "source" person.**

**`DeepFaceLab`**: <ins/>The current industry standard for high-fidelity swaps using autoencoders.</ins>  

**`FaceSwap`**: <ins/>An accessible, popular tool utilizing GAN-based refinement for smoother blending.</ins>  

**`FSGAN`**: <ins/>A model that enables swapping without needing to train on specific image pairs.</ins>  

---
#### ![](https://img.shields.io/badge/Face_Reenactment-4682B4?style=flat-square)
**The process of transferring motion, expressions, or speech patterns while keeping the target's identity.**

**`FOMM`**: <ins/>Uses a dense motion field to transfer expressions from a driving video to a source image.</ins>

**`3DMM`**: <ins/>Utilizes 3D geometry to manipulate facial mesh parameters for realistic changes.</ins>

**`Wav2Lip`**: <ins/>A specialized model ensuring generated lip movements match the audio input.</ins>

## Deep into MetaData

- [x] 원본영상_training_메타데이터.csv
- [x] 변조영상_training_메타데이터.csv
- [x] 원본영상_validation_메타데이터.csv
- [x] 변조영상_validation_메타데이터.csv

| Label | Train | Valid | Test | Total | 
| ----- | ----- | ----- | ---- | ----- |
| ![](https://img.shields.io/badge/FAKE-red?style=flat-square) | 139,951(79.50%) | 16,894(9.80%) | 18,971(10.70%) | 175,776 | 
| ![](https://img.shields.io/badge/REAL-blue?style=flat-square) | 49,419(79.62%) | 6,093(9.59%) | 6,654(10.79%) | 62,166 |


| Deepfake Model | Synthetic Video(Train) | Synthetic Video(Validation) |
|  ------------ | ------- | --------- |
|  `dfl` | 27,680 | 3,153 | 
|  `dffs` | 28,962 | 3,501 |
|  `fsgan` | 18,796 | 2,503 |
|  `fo`  | 49,586 |  5,853 |
|  `audio-driven` | 14,927 | 1,884 |
|  `total` | 139,951 | 16,894 | 

## How to Use Python Script

### 1. Split Data
This script generates train & test kodf metadata 

```python

python -m preprocess.kodf.split_data --root-dir DATA_ROOT --print-info True
```

> **Output**: train_metadata.csv, test_metadata.csv

| Column | Description |
| ------ | ----------- |
| `label` |  Ground truth label ('FAKE' or 'TRUE')          |
| `source` |  `original`, `fo`, `fsgan`, `dfl`, `dffs`, `audio-driven`   |
| `ori_vid` |  Video name used for manipulation        |
| `vid` |   Unique Video name       |

### 2. Detect Original Face

To maximize preprocessing efficiency, face detection is performed only on original (real) videos. Since manipulated videos in kodf share the same spatial coordinates as their sources, these bounding boxes are reused for the corresponding deepfake versions.

🚀 Efficiency Optimizations
- **Lightweight Model**: Uses yolov8n-face for high-speed inference without sacrificing accuracy.

- **Targeted Processing**: By detecting faces only in original videos, the total detection workload is reduced by approximately 80%.

- **Rescaling**: To Speed up for Training and inference, We  did x0.5 rescaling for all video frame:

```python
python -m preprocess.kodf.detect_original_face \
	--root-dir DATA_ROOT \
	--work-dir \
	--conf-thres 0.5 \
	--min-face-ratio 0.01 \
```

| Argument | Default | Description |
| -------- | ------- | ----------- |
| `--root-dir` | (Required)  | Root directory of the kodf dataset |
| `--work-dir` | (Required) | "Directory containing the video files to process" |
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
python -m preprocess.kodf.crop_face \
    --root-dir DATA_ROOT \
    --work-dir \
    --margin-ratio 0.2 \
    --margin-jitter 0.0 \
    --num-workers 1 
```
| Argument | Default | Description | 
| -------- | ------- | ----------- |
| `--work-dir` | Required | "Directory containing the video files to process" |
| `--margin-ratio` | 0.2 | Base padding ratio around the detected bounding box |
| `--margin-jitter` | 0.0 | Intenstiy of random noise added to the margin for each crop | 
| `--num-workers` | 1 | Number of Workers | 


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

_**`Frame Selection`**_: <ins/> Random 1 frame index per video (Range: 5s ~ 15s) </ins>

_**`Consistency`**_: <ins/> Same frame index applied to both Real and corresppnding Fake videos</ins>

_**`Face Quality`**_: <ins/>`min_face_ratio: 0.01` (to exclude tiny/low-quality faces)</ins>

_**`Crop Settings`**_: <ins/>`margin_ratio: 0.2` with `margin_jitter: 0`</ins>

| Category | source | Video Count | Extracted Frames | Detection Rate |
| :--- | :--- | :---: | :---: | :---: |
| **Real** | Original  | 49,419 | **49,404** | 99.97% |
| **Fake** | FO | 49,586 | 49,321 | 99.47% |
| | DFFS | 28,962 | 28,659 | 98.95% |
| | DFL | 27,680 | 27,393 | 98.96% |
| | FSGAN | 18,796 | 18,742 | 99.71% |
| | Audio-driven | 14,927 | 10,571 | 70.82% |
| **Total Fake** | **All Methods Combined** | **139,951** | **134,686** | **96.24%** |
