## 딥페이크 탐지 프로젝트

👉 You can also read the English version [here](README.md)

![MY Image](Images/deepfake.JPG)

(image from internet)

## 목차

- [설치 및 요구사항](#-설치-및-요구사항)
- [설정](#설정)
- [벤치마크 데이터셋](#-벤치마크-데이터셋)
- [전처리 파이프라인](#⚙️-전처리-파이프라인)
- [모델 구조](#🧠-모델-구조)
- [모델 평가](#📊-모델-평가)
- [저자](#📬-저자)
- [감사의 글](#🔗-감사의-글)
- [참고문헌](#📝-참고문헌)

## 💻 설치 및 요구사항

To install requirements: 

```
pip install -r requirements.txt
```

## 설정

Clone the repository and move into it:
```
git clone https://github.com/HanMoonSub/DeepGuard.git

cd DeepGuard
```

## 📦 벤치마크 데이터셋
- **DFFD**: 얼굴 조작 탐지용 데이터셋 [논문](http://cvlab.cse.msu.edu/pdfs/dang_liu_stehouwer_liu_jain_cvpr2020.pdf) [다운로드](http://cvlab.cse.msu.edu/dffd-dataset.html)

- **Celeb-DF(v2)**: 대규모 딥페이크 포렌식 데이터셋 [논문](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf) [다운로드](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v2)

- **DFDC**: DeepFake Detection Challenge(DFDC) 데이터셋 [논문](https://arxiv.org/abs/2006.07397) [다운로드](https://www.kaggle.com/c/deepfake-detection-challenge/data)

- **Celeb-DF++**: 일반화 성능 평가용 대규모 도전적 딥페이크 영상 데이터셋 [논문](https://arxiv.org/abs/2507.18015) [다운로드](https://github.com/OUC-VAS/Celeb-DF-PP)

- **Deeper Forensics-1.0**: 실제 환경 얼굴 위조 탐지용 대규모 데이터셋 [논문](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_DeeperForensics-1.0_A_Large-Scale_Dataset_for_Real-World_Face_Forgery_Detection_CVPR_2020_paper.pdf) [다운로드](https://github.com/EndlessSora/DeeperForensics-1.0)

- **ForgeryNet**: 종합 위조 분석을 위한 벤치마크 [논문](https://arxiv.org/abs/2103.05630) [다운로드](https://github.com/yinanhe/forgerynet)

- **IDForge**: 정체성 기반 멀티미디어 위조 데이터셋 [논문](https://arxiv.org/abs/2401.11764) [다운로드](https://github.com/xyyandxyy/IDForge)

| 데이터셋                  | 실제 영상 | 위조 영상 | 연도 | 설명                                                       |
| ------------------------ | ----------: | ----------: | ---: | ---------------------------------------------------------- |
| **DFFD**                 |      1,000 |     3,000 | 2019 | 다양한 SOTA 얼굴 조작 기법 포함                            |
| **Celeb-DF(v2)**         |         590 |       5,639 | 2019 | FaceForensics++ 개선, 고품질 영상                           |
| **DFDC**                 |      23,564 |     104,500 | 2019 | Kaggle 대회용, 대규모 다양성                                |
| **Deeper Forensics-1.0** |      50,000 |      10,000 | 2020 | 현실적 영상 조건 반영(압축, 블러 등)                        |
| **ForgeryNet**           |      99,630 |   121,617 | 2021 | 멀티모달 위조(영상+이미지) 벤치마크                         |
| **IDForge**              |       79,827 |      169,311 | 2024 | 동일 인물 기반 정체성 위조 데이터셋                          |
| **Celeb-DF++**           |         590 |      53,196 | 2025 | 일반화 성능 중심, 대규모 도전적 데이터셋                   |

## ⚙️ 전처리 파이프라인

## 🧠 모델 구조

## 📊 모델 평가



## 📬 저자
- 한문섭  
- 이예솔  
- 서윤제

## 🔗 감사의 글

## 📝 참고문헌

