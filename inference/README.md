# 🕵️‍♂️ Inference Pipeline & Strategy
The inference pipeline of this project is designed for real-world deepfake detection scenarios. Our strategy focuses on **capturing subtle forgery artifacts while ensuring highly stable and reliable predictions.**

## 🎯 Label

Label Mapping: **`Fake: 1`** | **`Real: 0`**

In the deepfake detection domain, misclassifying a fake video as real (False Negative) poses a significantly greater threat than misclassifying a real video as fake (False Positive).

## Test Time Augmentation(TTA)

Deepfake artifacts often consist of subtle, pixel-level manipulations. Applying heavy augmentation can inadvertently destroy these crucial forensic clues. So We applied a **`simple Horizontal Flip (p=0.5).`**

## Number of Frames & Face Crop

To ensure consistent and fair evaluation across diverse benchmarks (including Celeb-DF-v2, KoDF, and FaceForensics++), we standardized the frame extraction and face cropping parameters during inference:

- **`Fixed Number of Frames`**: We uniformly sample exactly `20 frames` per video. This guarantees a balanced computational load while maintaining sufficient temporal coverage across all datasets.

- **`Cropping Margin Ratio`**: A `20% margin` (margin_ratio=0.2) is applied around the detected face bounding box. This optimal margin ensures that blending boundaries and facial contours—crucial areas for deepfake artifacts—are fully preserved without including excessive background noise.

## Frame Aggregation Strategy

To derive a final video-level prediction, aggregating frame-level scores is essential. We designed a Custom Heuristic Method to overcome the critical blind spots of conventional aggregation techniques.

| Method | Features | Weakness |
| ------ | -------- | -------- | 
| **Mean Pooling** | Simple Average of all frame predictions | Highly vulnerable to partial deepfakes |
| **Top-K Pooling** | Average of the top-k highest predictions| Prone to malfuniction due to single-frame noise or outliers |
| **Custom Heuristic** | Conditional probability averaging| Highly robust against both fully and partially manipulated videos |

💡 **Heuristic Aggregation Logic**

```python
import numpy as np

def heuristic_aggregation(preds: np.ndarray) -> float:
    sz = len(preds)
    
    t_fake = 0.8  # Threshold for strong Fake confidence
    t_real = 0.2  # Threshold for strong Real confidence
    
    fakes = preds[preds > t_fake]
    num_fakes = len(fakes)
    
    # Condition 1: If ~40% or more of the frames are predicted as strong Fakes
    if num_fakes > (sz // 2.5):
        return float(np.mean(fakes))
        
    # Condition 2: If over 90% of the frames are confidently Real
    elif np.count_nonzero(preds < t_real) > 0.9 * sz:
        return float(np.mean(preds[preds < t_real]))
        
    # Condition 3: Ambiguous cases
    else:
        return float(np.mean(preds))
```
## 📚 DeepFake Video Benchmark Datasets

To evaluate the generalization and robustness of our deepfake detection model, we utilize three large-scale, widely recognized benchmark datasets. Each dataset presents unique challenges and covers different types of forgery methods.

| Dataset | Real Videos | Fake Videos | Year | Participants | Description (Paper Title) | Details |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Celeb-DF-v2** | 890 | 5,639 | 2019 | 59 | *A Large-scale Challenging Dataset for DeepFake Forensics* | [🔗 Readme](../preprocess/celeb_df_v2/README.md) |
| **FaceForensics++** | 1,000 | 6,000 | 2019 | 1,000 | *Learning to Detect Manipulated Facial Images* | [🔗 Readme](../preprocess/ff++/README.md) |
| **KoDF** | 62,166 | 175,776 | 2020 | 400 | *Large-Scale Korean Deepfake Detection Dataset* | [🔗 Readme](../preprocess/kodf/README.md) |

> **Note:** For detailed information on data preprocessing, frame extraction, and formatting for each dataset, please refer to the respective `Readme` links in the table above.

## 🎬 Celeb DF(V2) Video Evaluation

<details>
<summary><span style="font-size: 1.25em; font-weight: bold;">✅ Test Data Evaluation</span></summary>
 
| 🤖 Model | 🖼️ Frames | ⚙️ Agg Mode | 🔄 TTA (H-Flip) | 🎯 Accuracy | 📈 AUC | 📉 Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.9742 | 0.9877 | 0.0625 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.9900 | 0.9900 | 0.0408 |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.9842 | 0.9965 | 0.0283 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.9981** | **0.9984** | **0.0089** |

</details>

<details>
<summary><span style="font-size: 1.25em; font-weight: bold;">🔗 Cross Evaluation</span></summary>

#### 📌 Dataset: `celeb_df_v2` | 🧠 Model: `ff++`

| 🤖 Model | 🖼️ Frames | ⚙️ Agg Mode | 🔄 TTA (H-Flip) | 🎯 Accuracy | 📈 AUC | 📉 Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.7317 | 0.6949 | 0.6816 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.7143 | 0.7266 | 0.7662 |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.7259 | 0.6999 | 0.6794 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.7722** | **0.7309** | **0.6657** |


#### 📌 Dataset: `celeb_df_v2` | 🧠 Model: `kodf`

| 🤖 Model | 🖼️ Frames | ⚙️ Agg Mode | 🔄 TTA (H-Flip) | 🎯 Accuracy | 📈 AUC | 📉 Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.5579 | 0.4719 | 1.2605 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.5946** | **0.5400** | **1.0078** |
</details>

## 🎬 FaceForensics++ Video Evaluation

<details>
<summary><span style="font-size: 1.25em; font-weight: bold;">✅ Test Data Evaluation</span></summary>
 
| 🤖 Model | 🖼️ Frames | ⚙️ Agg Mode | 🔄 TTA (H-Flip) | 🎯 Accuracy | 📈 AUC | 📉 Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.9517 | 0.9860 | 0.1334 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.9842 | **0.9977** | **0.0477** |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.9808 | 0.9969 | 0.0637 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.9850** | 0.9974 | 0.0492 |

</details>

<details>
<summary><span style="font-size: 1.25em; font-weight: bold;">🔗 Cross Evaluation</span></summary>

#### 📌 Dataset: `ff++` | 🧠 Model: `celeb_df_v2`

| 🤖 Model | 🖼️ Frames | ⚙️ Agg Mode | 🔄 TTA (H-Flip) | 🎯 Accuracy | 📈 AUC | 📉 Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.5241 | 0.6860 | 1.1241 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.5533 | 0.7299 | 0.9987 |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.5492 | 0.7301 | 1.0556 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.5825** | **0.7307** | **0.8897** |


#### 📌 Dataset: `ff++` | 🧠 Model: `kodf`

| 🤖 Model | 🖼️ Frames | ⚙️ Agg Mode | 🔄 TTA (H-Flip) | 🎯 Accuracy | 📈 AUC | 📉 Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | **0.4875** | 0.5341 | 1.6178 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | 0.4525 | **0.5902** | **1.5321** |
</details>

## Kodf Video Evaluation 
