# Deepfakes Detection (딥페이크 탐지)

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
  <a href="README.md"><b>🇺🇸 English Version</b></a> | 
  <a href="#-모델-평가"><b>📈 모델 평가</b></a> | 
  <a href="#-이미지-및-비디오-예측"><b>🔮 데모 실행</b></a>
</p>

## 📌 목차

- [💡 설치 및 요구사항](#-설치-및-요구사항)
- [🛠 설정](#-설정)
- [📚 딥페이크 비디오 벤치마크 데이터셋](#-딥페이크-비디오-벤치마크-데이터셋) — 학습에 사용된 Celeb-DF-v2, FF++, KoDF 데이터셋 개요.
- [⚙️ 데이터 준비](#데이터-준비) — YOLOv8을 이용한 효율적인 얼굴 검출 및 랜드마크 추출 파이프라인.
- [🏗 모델 구조](#-모델-구조) — 하이브리드 CNN-ViT (MS-EffViT & MS-EffGCViT) 설계 상세.
- [🧬 모델 주(Model Zoo)](#-모델-주model-zoo) — 모델 변체별 파라미터 수 및 연산량(FLOPs) 비교.
- [🚀 학습](#-학습) - Google Colab 및 W&B를 활용한 단계별 학습 스크립트.
- [📈 모델 평가](#-모델-평가) - 벤치마크 결과.
- [💻 모델 사용법](#-모델-사용법) - Python 코드 및 timm을 통한 DeepGuard 모델 통합 방법.
- [🔮 이미지 및 비디오 예측](#-이미지-및-비디오-예측) - 딥페이크 탐지를 위한 간단한 추론 예시.
- [📬 제작자](#-제작자)
- [📝 참고 문헌](#-참고-문헌)
- [⚖️ 라이선스](#-라이선스)

---

## 💡 설치 및 요구사항

필수 라이브러리 설치:

```bash
pip install -r requirements.txt
```

## 🛠 설정

저장소를 클론하고 해당 디렉토리로 이동합니다:
```bash
git clone [https://github.com/HanMoonSub/DeepGuard.git](https://github.com/HanMoonSub/DeepGuard.git)
cd DeepGuard
```

## 📚 딥페이크 비디오 벤치마크 데이터셋

모델의 범용성과 강건성을 평가하기 위해 널리 인정받는 세 가지 대규모 벤치마크 데이터셋을 사용합니다. 각 데이터셋은 서로 다른 조작 기법과 도전적인 과제들을 포함하고 있습니다.

| 데이터셋 | 실제 영상 | 위조 영상 | 연도 | 참여 인원 | 설명 (논문 제목) | 상세 정보 |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Celeb-DF-v2** | 890 | 5,639 | 2019 | 59 | *A Large-scale Challenging Dataset for DeepFake Forensics* | [🔗 Readme](preprocess/celeb_df_v2/README.md) |
| **FaceForensics++** | 1,000 | 6,000 | 2019 | 1,000 | *Learning to Detect Manipulated Facial Images* | [🔗 Readme](preprocess/ff++/README.md) |
| **KoDF** | 62,166 | 175,776 | 2020 | 400 | *Large-Scale Korean Deepfake Detection Dataset* | [🔗 Readme](preprocess/kodf/README.md) |
 
<div id="데이터-준비"></div>

## ⚙️ 데이터 준비

전처리 파이프라인은 비디오에서 얼굴 특징을 효율적으로 추출하여 고정밀 딥페이크 탐지를 준비하도록 설계되었습니다.

### 원본 얼굴 검출 (Detect Original Face)
전처리 효율을 극대화하기 위해, <ins>**얼굴 검출은 원본(Real) 비디오에서만 수행됩니다.**</ins> 벤치마크 데이터셋의 조작된 영상들은 원본 영상과 동일한 공간 좌표를 공유하므로, 추출된 바운딩 박스를 위조 영상에도 그대로 재사용합니다.

🚀 **효율화 최적화**
- **경량 모델**: 정확도를 유지하면서 빠른 추론 속도를 보장하는 `yolov8n-face`를 사용합니다.
- **타겟 프로세싱**: 원본 영상에서만 얼굴을 검출함으로써 전체 검출 작업량을 약 80% 줄였습니다.
- **동적 리사이징**: 다양한 해상도에서 일정한 추론 속도를 유지하기 위해 프레임 크기에 따라 자동으로 크기를 조절합니다:

| 프레임 크기(긴 변 기준) | 스케일 팩터 | 동작 |
| ------------------------ | ------------ | ------ |
|         < 300px          |      2.0     |  ![](https://img.shields.io/badge/Upscale-green?style=flat-square) |
|      300px - 700px       |      1.0     | ![](https://img.shields.io/badge/No_Change-gray?style=flat-square) |
|      700px - 1500px      |      0.5     | ![](https://img.shields.io/badge/DownScale-skyblue?style=flat-square) |
|          > 1500px        |      0.33    | ![](https://img.shields.io/badge/DownScale-skyblue?style=flat-square) | 

### 얼굴 크롭 및 랜드마크 추출
이 모듈은 이전 단계에서 생성된 바운딩 박스를 사용하여 원본 및 위조 영상 모두에서 얼굴 영역을 크롭합니다. 또한, Landmark-based Cutout과 같은 고급 증강 기법을 위해 랜드마크 검출을 수행합니다.

🛠 **주요 기능**
- **지터링을 포함한 동적 마진**: 얼굴 주변에 설정 가능한 마진을 추가합니다. `margin_jitter` 파라미터는 크롭 크기에 무작위 변화를 주어 모델이 다양한 얼굴 스케일에 강건해지도록 돕습니다.
- **랜드마크 지역화**: 5개의 주요 얼굴 랜드마크(눈, 코, 입꼬리)를 감지하고 `.npy` 파일로 저장합니다.

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

### 데이터셋별 파이프라인

각 데이터셋의 구체적인 전처리 세부 사항을 확인하려면 아래 링크를 클릭하세요:

* [**Celeb-DF V2 전처리**](/preprocess/celeb_df_v2/README.md)
* [**FaceForensics++ 전처리**](/preprocess/ff++/README.md)
* [**KoDF 전처리**](/preprocess/kodf/README.md)

## 🏗 모델 구조

**Multi Scale Efficient Global Context Vision Transformer**는 최적화된 멀티 스케일 하이브리드 아키텍처입니다. CNN의 공간적 귀납 편향(Spatial Inductive Bias)과 계층적 어텐션 메커니즘을 통합하여, 정교한 탐지를 위해 미세한(Local) 아티팩트와 거시적인(Global) 아티팩트를 효과적으로 식별합니다.

#### 상세 정보 확인

- [모델 구조: MS-EffViT](deepguard/MS_EffViT.md) - _**Multi Scale Efficient Vision Transformer**_
- [심화 구조: MS-EFFGCViT](deepguard/MS_EffGCViT.md) - _**Multi Scale Efficient Global Context Vision Transformer**_


<p align="center">
  <img src="https://raw.githubusercontent.com/HanMoonSub/DeepGuard/main/Images/ms_eff_gcvit.JPG" width="100%" height="700">
</p>

특징 맵 전체에 걸쳐 장거리(Long-range) 및 단거리(Short-range) 정보를 모두 캡처하기 위해 두 가지 유형의 셀프 어텐션을 활용합니다.

- **Local Window Attention**: 이미지 크기에 비례하는 선형 계산 복잡도를 유지하면서 국부적인 질감과 정밀한 공간적 세부 사항을 효율적으로 캡처합니다.
- **Global Window Attention**: Swin Transformer와 달리, 이 모듈은 로컬 윈도우의 Key, Value와 상호작용하는 글로벌 쿼리(Global-queries)를 사용합니다. 이를 통해 각 로컬 영역이 전역 컨텍스트를 수용하게 함으로써 장거리 의존성을 효과적으로 파악하고 전체 공간 구조에 대한 포괄적인 이해를 제공합니다.

<p align="center">
  <img src=https://raw.githubusercontent.com/HanMoonSub/DeepGuard/main/Images/window_attention.JPG width="100%" height="300">
</p>

## 🧬 모델 주(Model Zoo)

| 모델명 | 해상도 | 총 파라미터(M) | 백본(M) | L-ViT(M) | H-ViT(M) | 연산량(FLOPs, G) | 설정 파일 |
| ----- | ---------- | -------------- | ----------- |------------- | ------------- | --------------  | ------- | 
| ⚡ ms_eff_gcvit_b0 | 224 X 224 | 8.7 | 3.6(41.4%) | 1.7(19.5%) | 3.3(37.9%) | 0.87 | [spec](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |
| 🔥 ms_eff_gcvit_b5 | 384 X 384 | 50.3 | 27.3(54.3%) | 6.6(13.1%) | 16.1(32.0%) | 13.64 | [spec](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

## 🚀 학습

`ms_eff_vit` 및 `ms_eff_gcvit` 모두에 대한 학습 스크립트를 제공합니다. 무료 GPU 환경을 위해 **Google Colab**을, 실험 기록 및 트래킹을 위해 **Weights & Biases(W&B)** 사용을 권장합니다.

#### 📊 Weight & Biases 실험 결과

* **ms_eff_vit_b0:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_vit_b0_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_vit_b0_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_vit_b0_kodf)
* **ms_eff_vit_b5:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_vit_b5_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_vit_b5_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_vit_b5_kodf)
* **ms_eff_gcvit_b0:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b0_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b0_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b0_kodf)
* **ms_eff_gcvit_b5:** [Celeb-DF-v2 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b5_celeb_df_v2) | [FaceForensics++ 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b5_ff++) | [KoDF 🚀](https://wandb.ai/origin6165/ms_eff_gcvit_b5_kodf)

```python
!python -m train_eff_vit \ # 또는 train_eff_gcvit
    --root-dir DATA_ROOT \ 
    --model-ver "ms_eff_vit_b5" \ # ms_eff_vit_b0, ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5
    --dataset "ff++" \ # ff++, celeb_df_v2, kodf
    --seed 2025 \ # 재현성을 위한 시드값
    --wandb-api-key "사용자-API-키" # 본인의 W&B API 키를 입력하세요
```

## 📈 모델 평가

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

**Celeb DF(v2) 사전 학습 모델**

| 모델 버전 | Test@Acc | Test@Auc | Test@log_loss | 다운로드 | 학습 레시피 |
| ------------- | -------- | -------- | ---------- | -------- | ----- |
| ms_eff_gcvit_b0 | 0.9842 | 0.9965 | 0.0283 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b0_celeb_df_v2.bin) | [recipe](deepguard/config/ms_eff_gcvit_b0/celeb_df_v2.yaml) |
| ms_eff_gcvit_b5 | 0.9981 | 0.9984 | 0.0089 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b5_celeb_df_v2.bin) | [recipe](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

**FaceForensics++ 사전 학습 모델**

| 모델 버전 | Test@Acc | Test@Auc | Test@log_loss | 다운로드 | 학습 레시피 |
| ------------- | -------- | -------- | ---------- | -------- | ------ |
| ms_eff_gcvit_b0 | 0.9808 | 0.9969 | 0.0637| [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b0_ff++.bin) | [recipe](deepguard/config/ms_eff_gcvit_b0/celeb_df_v2.yaml) |
| ms_eff_gcvit_b5 | 0.9850 | 0.9974 | 0.0492 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_gcvit_b5_ff++.bin) | [recipe](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

**KoDF 사전 학습 모델**

| 모델 버전 | Test@Acc | Test@Auc | Test@log_loss | 다운로드 | 학습 레시피 |
| ------------- | -------- | -------- | ---------- | -------- | ------ |
| ms_eff_gcvit_b0 | 0.9655 | 0.9792 | 0.1237 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.2.0/ms_eff_gcvit_b0_kodf.bin) | [recipe](deepguard/config/ms_eff_gcvit_b0/celeb_df_v2.yaml) |
| ms_eff_gcvit_b5 | 0.9850 | 0.9974 | 0.0492 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.2.0/ms_eff_gcvit_b5_kodf.bin) | [recipe](deepguard/config/ms_eff_gcvit_b5/celeb_df_v2.yaml) |

## 💻 모델 사용법

**빠른 시작**
`DeepGuard` 패키지를 직접 임포트하거나 `timm` 인터페이스를 통해 모델을 로드할 수 있습니다.

**지원 데이터셋**: `celeb_df_v2`, `ff++`, `kodf`

**설치**

```bash
# pip install -U git+[https://github.com/HanMoonSub/DeepGuard.git](https://github.com/HanMoonSub/DeepGuard.git)
pip install deepguard
```


**방법 A: 직접 임포트 (DeepGuard 사용)**

```python
from deepguard import ms_eff_gcvit_b0, ms_eff_gcvit_b5

model = ms_eff_gcvit_b0(pretrained=True, dataset="celeb_df_v2")
model = ms_eff_gcvit_b5(pretrained=True, dataset="ff++")
```

**방법 B: timm 인터페이스 사용**

```python
import timm
import deepguard

model = timm.create_model("ms_eff_gcvit_b0", pretrained=True, dataset="ff++")
model = timm.create_model("ms_eff_gcvit_b5", pretrained=True, dataset="kodf")
```

## 🔮 이미지 및 비디오 예측

#### 딥페이크 이미지 예측

```python
from inference.image_predictor import ImagePredictor

# 예측기 초기화
predictor = ImagePredictor(
            margin_ratio = 0.2, # 검출된 얼굴 크롭 주변의 마진 비율
            conf_thres = 0.5, # 얼굴 검출 신뢰도 임계값
            min_face_ratio = 0.01, # 처리를 위한 프레임 대비 최소 얼굴 크기 비율
            model_name = "ms_eff_vit_b0", # ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5 중 선택
            dataset = "celeb_df_v2" # ff++, kodf 데이터셋 중 선택
            )

# 추론 실행
result = predictor.predict_img(
            img_path="path/to/image.jpg",
            tta_hflip=0.0 # 테스트 시 증강(TTA)을 위한 수평 뒤집기 확률
            )

print(f"딥페이크 확률: {result:.4f}")
```

#### 딥페이크 비디오 예측

```python
from inference.video_predictor import VideoPredictor

# 예측기 초기화
predictor = VideoPredictor(
            margin_ratio = 0.2, # 검출된 얼굴 크롭 주변의 마진 비율
            conf_thres = 0.5, # 얼굴 검출 신뢰도 임계값
            min_face_ratio = 0.01, # 처리를 위한 프레임 대비 최소 얼굴 크기 비율
            model_name = "ms_eff_vit_b0", # ms_eff_vit_b5, ms_eff_gcvit_b0, ms_eff_gcvit_b5 중 선택
            dataset = "celeb_df_v2" # ff++, kodf 데이터셋 중 선택
            )

# 추론 실행
result = predictor.predict_video(
            video_path = "path/to/video.mp4",
            num_frames = 20, # 비디오당 샘플링할 프레임 수
            agg_mode = "conf", # 집계 방식: 'conf', 'mean', 'vote'
            tta_hflip=0.0 # 테스트 시 증강(TTA)을 위한 수평 뒤집기 확률
            )

print(f"딥페이크 확률: {result:.4f}")
```


## 📬 제작자

_**본 프로젝트는 충북대학교(CBNU) 소프트웨어학부 졸업 작품(Senior Graduation Project)으로 개발되었습니다.**_

* **한문섭**: **Data & Backend Engineering** (데이터 전처리 파이프라인, DB 스키마 설계) — [hanmoon3054@gmail.com](mailto:hanmoon3054@gmail.com)
* **이예솔**: **UI/UX & Frontend Engineering** (UI/UX 디자인, 사용자 대시보드, 모델 시각화) — [yesol4138@chungbuk.ac.kr](mailto:yesol4138@chungbuk.ac.kr)
* **서윤제**: **AI Engineering** (AI 모델 구조 설계, 추론 API 설계, 모델 서빙) — [seoyunje2001@gmail.com](mailto:seoyunje2001@gmail.com)


## 📝 참고 문헌

1. [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) - _Tim Esler의 사전 학습된 얼굴 검출(MTCNN) 및 인식(InceptionResNet) 모델_
2. [`face-cutout`](https://github.com/sowmen/face-cutout) - _Sowmen의 Face Cutout 라이브러리_
3. [`Celeb-DF++`](https://github.com/OUC-VAS/Celeb-DF-PP) - _OUC-VAS Group의 Celeb-DF++ 데이터셋_
4. [`DeeperForensics-1.0`](https://github.com/EndlessSora/DeeperForensics-1.0) - _Endless Sora의 DeeperForensics-1.0 데이터셋_
5. [`Deepfake Detection`](https://github.com/abhijithjadhav/Deepfake_detection_using_deep_learning) - _Abhijith Jadhav의 ResNext와 LSTM을 이용한 딥페이크 탐지_
6. [`deepfake-detection-project-v4`](https://github.com/ameencaslam/deepfake-detection-project-v4) - _Ameen Caslam의 다양한 딥러닝 모델들_
7. [`Awesome-Deepfake-Detection`](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection
) - _Daisy Zhang이 정리한 도구, 논문 및 코드 리스트_

## ⚖️ 라이선스 

본 프로젝트는 MIT 라이선스의 규정에 따라 라이선스가 부여됩니다.