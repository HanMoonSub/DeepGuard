# 🚀 Multi Scale Efficient Vision Transformer

This Repository presents the PyTorch implementation of **Multi Scale Efficient Vision Transformer**, a hybrid architecture optimized for **deepfake detection task**

This model is a **frame-level** and **spatial-domain** architecture, designed to perform classification tasks on both **static images** and **video sequences**

## 💥 News 💥

- [**02.03.2026**] 🔥🔥 We have released **FaceForensics++** fine-tuned **MS-Eff-ViT B5** model weightes for **384X384**
- [**02.03.2026**] 🔥🔥 We have released **Celeb DF(V2)** fine-tuned **MS-Eff-ViT B5** model weightes for **384X384**
- [**02.03.2026**] 🔥 We have released **FaceForensics++** fine-tuned **MS-Eff-ViT B0** model weightes for **224X224**
- [**02.03.2026**] 🔥 We have released **Celeb DF(V2)** fine-tuned **MS-Eff-ViT B0** model weightes for **224X224**

## Model Architecture

## Model Performance

- Metric & FLOPS

- Metric & Params

## 📊 Model Zoo

| Model | Resolution | # Total Params(M) | # Backbone(M) | # L-ViT(M) | # H-ViT(M)  | FLOPs (G) | Model Config |
| ----- | ---------- | ---------- | ------------------------- | ------------- | --------------  | ------- | ------- |
| ⚡ ms_eff_vit_b0 | 224 X 224 | 5.9 | 3.6(61%) | 0.5(8.5%) | 1.7(29%) | 0.68 | [spec](./config/ms_eff_vit_b0/celeb_df_v2.yaml) |
| 🔥 ms_eff_vit_b5 | 384 X 384 | 52.0 | 27.3(52.5%) | 4.7(9%) | 19.7(37.9%) | 15.22 | [spec](./config/ms_eff_vit_b5/celeb_df_v2.yaml) |

## 🛠 Model Variants 

**⚡ ms_eff_vit_b0 (Fast Mode / Mobile)**: Efficiency at the Edge
- Optimized for **real-time inference** and mobile deployment.

**🔥 ms_eff_vit_b5 (Pro Mode / Enterprise)**: Uncompromising Precision
- Engineered for high-fidelity analysis and enterprise-grade accuracy.

## ⚙️ Model Weight Initialization
The model incorporates a hybrid initialization strategy to leverage pre-trained features while ensuring stable convergence of the transformer components

> **Backbone**: **ImageNet-1K Pretraiend Weights**(EfficientNet)  

> **L-ViT / H-ViT / Head**: Truncated Normal( `std=0.02` )
 
> **No Weight Decay**: `pos-embed`, `cls-token`, `bn`, `norm`

## DeepFake Video Benchmarks

🔥 **Celeb-DF(v2)**: A Large-scale Challenging Dataset for DeepFake Forensics [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.pdf) [download](https://github.com/yuezunli/celeb-deepfakeforensics/tree/master/Celeb-DF-v2)

🔥 **FaceForensics++**: Learning to Detect Manipulated Facial Images [paper](https://arxiv.org/abs/1901.08971) [download](https://github.com/ondyari/FaceForensics)


**Celeb DF(v2) Pretrained Models**

| Model Variant | Test@Acc | Test@Auc | Test@log_loss | Download | Train Config |
| ------------- | -------- | -------- | ---------- | -------- | ----- |
| ms_eff_vit_b0 | 0.9842 | 0.9877 | 0.0625 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_vit_b0_celeb_df_v2.bin) | [recipe](./config/ms_eff_vit_b0/celeb_df_v2.yaml) |
| ms_eff_vit_b5 | 0.9900 | 0.9900 | 0.0408 |[model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_vit_b5_celeb_df_v2.bin) | [recipe](./config/ms_eff_vit_b5/celeb_df_v2.yaml) |

**FaceForensics++ Pretrained Models**

| Model Variant | Test@Acc | Test@Auc | Test@log_loss | Download | Train Config |
| ------------- | -------- | -------- | ---------- | -------- | ------ |
| ms_eff_vit_b0 | 0.9517 | 0.9860 | 0.1334 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_vit_b0_ff++.bin) | [recipe](./config/ms_eff_vit_b0/celeb_df_v2.yaml) |
| ms_eff_vit_b5 | 0.9842 | 0.9977 | 0.0477 | [model](https://github.com/HanMoonSub/DeepGuard/releases/download/v0.1.0/ms_eff_vit_b5_ff++.bin) | [recipe](./config/ms_eff_vit_b5/celeb_df_v2.yaml) |


## Usage

**Quick Start**
You can load the models directly via the `DeepGuard` package or through the `timm` interface.

**Available Datasets**: `celeb_df_v2`, `ff++`

**Installation**

```bash
pip install -U git+https://github.com/HanMoonSub/DeepGuard.git
```


**Option A: Direct Import (via DeepGuard)**

```python
from deepguard import ms_eff_vit_b0, ms_eff_vit_b5

model = ms_eff_vit_b0(pretrained=True, dataset="celeb_df_v2")
model = ms_eff_vit_b5(pretrained=True, dataset="ff++")
```

**Option B: Using timm Interface (via timm)**

```python
import timm
import deepguard

model = timm.create_model("ms_eff_vit_b0", pretrained=True, dataset="celeb_df_v2")
model = timm.create_model("ms_eff_vit_b5", pretrained=True, dataset="ff++")
```