# 🧩 Data Preprocessing Pipeline

The preprocessing pipeline for detecting, cropping, and preparing face data from the DeepFake Detection dataset.

## 🧠 Overview

| Step | Script                                                   | Description                                                                |
| ---- | -------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1️⃣  | [`detect_original_faces.py`](#1️⃣-detect-original-faces) | Detect faces **only from original videos** and save bounding boxes as JSON |
| 2️⃣  | [`extract_crops.py`](#2️⃣-extract-cropped-faces)         | Extract **cropped face images** from both original & fake videos           |
| 3️⃣  | [`generate_landmarks.py`](#3️⃣-generate-landmarks)       | Generate **facial landmarks (.npy)** for cropped faces                     |
| 4️⃣  | [`generate_metadata.py`](#4️⃣-generate-metadata)         | Create **metadata CSV** for the cropped dataset                             |


## 1️⃣ Detect Original Faces

**`detect_original_faces.py`**  

Detects faces only from original videos to minimize redundant computation.
The resulting .json file stores bounding box coordinates for each frame.

🧭 Command
```
python -m Preprocessing.detect_original_faces \
  --root_dir DATA_ROOT \
  --detector_type "FacenetDetector" \
  --batch_size 32 \
  --apply_clahe True
```

⚙️ Arguments

| Argument          | Description                                                   | Default           |
| ----------------- | ------------------------------------------------------------- | ----------------- |
| `--root_dir`      | Root directory of the dataset                                 | *(required)*      |
| `--detector_type` | Type of face detector (`FacenetDetector`, `RetinaFace`, etc.) | `FacenetDetector` |
| `--batch_size`    | Batch size for detection with face detector                                     | `32`              |
| `--apply_clahe`   | Apply CLAHE enhancement before detection                      | `False`           |
           |


## 2️⃣ Extract Cropped Faces

**`extract_crops.py`**

Uses bounding boxes from the previous step to extract face crops (as .png images).
Applies to both original and fake videos.

🧭 Command
```
python -m Preprocessing.extract_crops \
  --root_dir DATA_ROOT \
  --crops_dir crops \
  --frame_interval 10 \
  --margine_ratio 0.3
```

⚙️ Arguments

| Argument           | Description                            | Default      |
| ------------------ | -------------------------------------- | ------------ |
| `--root_dir`       | Root directory of the dataset          | *(required)* |
| `--crops_dir`      | Output directory to save cropped faces | *(required)* |
| `--frame_interval` | Process every Nth frame                | `10`         |
| `--margine_ratio`  | Margin around face bounding box(0.3 -> 30%)        | `0.3`        |



## 3️⃣ Generate Landmarks

**`generate_landmarks.py`**

Detects facial landmarks (eyes, nose, mouth, etc.) using MTCNN and saves them as .npy.

🧭 Command
```
python -m Preprocessing.generate_landmarks \
  --root_dir DATA_ROOT \
```
⚙️ Arguments
| Argument     | Description                             | Default      |
| ------------ | --------------------------------------- | ------------ |
| `--root_dir` | Directory containing the `crops` folder | *(required)* |

📍 Useful for Dynamic CutOut Augmention for robust 

## 4️⃣ Generate Data Folds

**`generate_metadata.py`**

CSV metadata file containing video name, frame name, frame_idx, face_idx, label, ori_vid

📦 Command
```
python -m Preprocessing.generate_metadata \
  --root_dir DATA_ROOT \
  --output_dir outs \

```

⚙️ Arguments
| Argument           | Description                             | Default      |
| ------------------ | --------------------------------------- | ------------ |
| `--root_dir`       | Root directory of the dataset           | *(required)* |
| `--output_dir`      | Directory to save `metadata.csv`   | *(required)* |

## ✅ Final Output Structure

```

DATA_ROOT/
├── subfolder_0/
│   ├── video1.mp4
│   ├── metadata.json
│   └── ...
│
├── subfolder_1/
│   ├── video1.mp4
│   ├── metadata.json
│   └── ...
│
├── subfolder_2/
│   ├── video1.mp4
│   ├── metadata.json
│   └── ...
│
├── ... (more subfolders)
│
├── boxes/                     # 🧠 Face bounding boxes (only ORIGINAL videos)
│   ├── original_video1.json
│   ├── original_video2.json
│   └── ...
│
├── landmarks/                 # 📍 Face landmarks (only ORIGINAL videos)
│   ├── original_video1/
│   │   ├── 0_0.npy            # {frame_idx}_{face_idx}
│   │   ├── 10_0.npy
│   │   └── ...
│   ├── original_video2/
│   │   └── ...
│   └── ...
│
├── crops/                     # 🎞️ Cropped face images (ORIGINAL + FAKE videos)
│   ├── original_video1/
│   │   ├── 0_0.png            # {frame_idx}_{face_idx}
│   │   ├── 10_0.png
│   │   └── ...
│   ├── fake_video1/
│   │   ├── 0_0.png
│   │   └── ...
│   └── ...
│
└── outs/                      # 🗂️ Output files
    └── train_metadata.csv


```