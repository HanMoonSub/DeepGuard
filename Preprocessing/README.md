# 🧩 Data Preprocessing

This section describes the preprocessing pipeline for detecting and extracting faces from the original DeepFake videos.

---

## 1️⃣ Detect Original Faces

**`detect_original_faces.py`**  

Detect faces only from the *original* videos.  
This script generates a `.json` file containing bounding box coordinates for the detected faces in each frame.

📦 Command
```
python -m Preprocessing.detect_original_faces \
  --root_dir DATA_ROOT \
  --detector_type "FacenetDetector" \
  --device "cuda:0" \
  --batch_size 32
```

⚙️ Arguments
| Argument          | Description                       | Default           |
| ----------------- | --------------------------------- | ----------------- |
| `--root_dir`      | Root directory of the dataset     | *(required)*      |
| `--detector_type` | Type of face detector to use      | `FacenetDetector` |
| `--device`        | Device for running face detection | `cuda:0`          |
| `--batch_size`    | Batch size for face detection     | `32`              |

## 2️⃣ Extract Cropped Faces

**`extract_crops.py`**

Extract face regions from videos using the bounding boxes generated in the previous step, and save them as .png images.

📦 Command
```
python -m Preprocessing.extract_crops \
  --root_dir DATA_ROOT \
  --crops_dir crops \
  --frame_interval 10 \
  --margine_ratio 0.3
```

⚙️ Arguments
| Argument           | Description                             | Default      |
| ------------------ | --------------------------------------- | ------------ |
| `--root_dir`       | Root directory of the dataset           | *(required)* |
| `--crops_dir`      | Directory to save cropped face images   | *(required)* |
| `--frame_interval` | Interval between processed frames       | `10`         |
| `--margine_ratio`  | Margin ratio added around the face crop | `0.3`        |


## 3️⃣ Generate Landmarks

**`generate_landmarks.py`**

Identify the face landmarks using MTCNN face detector and save them as `.npy` files

## 4️⃣ Generate SSIM Diffs

**`generate_ssim_diffs.py`**

Generate and save the SSIM difference masks used for facecutout. This is also optional, but speeds up runtime calcuations.

## 5️⃣ Generate Data Folds

**`generate_data_folds.py`**

Create train/validation folds by seperating source faces. This ensures that there is no data leak for testing.