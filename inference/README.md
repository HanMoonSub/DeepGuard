# Inference Strategy

## Label

Label Mapping: **`Fake: 1`** | **`Real: 0`**

## Test Time Augmentation(TTA)

Deepfake artifacts often consist of subtle, pixel-level manipulations. Applying heavy augmentation can inadvertently destroy these crucial forensic clues. So We applied a **`simple Horizontal Flip (p=0.5).`** TTA is controlled by `tta_hflip` in `get_test_transforms` (`deepguard/data/transforms.py`):

```python
from deepguard.data import get_test_transforms

# Horizontal Flip TTA (p=0.5)
transforms = get_test_transforms(img_size=[224, 224], tta_hflip=0.5)
```

Internally, `get_test_transforms` resizes the image, appends `A.HorizontalFlip(p=tta_hflip)` only when `tta_hflip != 0`, and then applies normalization.

## Number of Frames & Face Crop

To ensure consistent and fair evaluation across diverse benchmarks (including **_Celeb-DF-v2_**, **_KoDF_**, and **_FaceForensics++_**), we standardized the frame extraction and face cropping parameters during inference:

**`Fixed Number of Frames`**: We uniformly sample exactly **`20 frames`** per video. This guarantees a balanced computational load while maintaining sufficient temporal coverage across all datasets.

**`Cropping Margin Ratio`**: A **`20% margin`** (margin_ratio=0.2) is applied around the detected face bounding box. This optimal margin ensures that blending boundaries and facial contours—crucial areas for deepfake artifacts—are fully preserved without including excessive background noise.

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


# In-domain Evaluation

Each model is trained and evaluated on the same dataset: it is tested on the held-out test split of the dataset it was trained on. 

### Celeb-DF-v2
 
| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.9742 | 0.9877 | 0.0625 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.9900 | 0.9900 | 0.0408 |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.9842 | 0.9965 | 0.0283 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.9981** | **0.9984** | **0.0089** |

### FaceForensics++ 
 
| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.9517 | 0.9860 | 0.1334 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.9842 | **0.9977** | **0.0477** |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.9808 | 0.9969 | 0.0637 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.9850** | 0.9974 | 0.0492 |

### KoDF 
 
| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.9655 | 0.9792 | 0.1237 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.9792** | **0.9831** | **0.0692** |


# Cross-domain Evaluation

Each model is trained on one dataset and evaluated on a different, unseen dataset. Each table is titled `Train → Test`: the dataset the weights were trained on, followed by the dataset used for evaluation.

### Celeb-DF-v2

### Train: `ff++` → Test: `celeb_df_v2`

| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.7317 | 0.6949 | 0.6816 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.7143 | 0.7266 | 0.7662 |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.7259 | 0.6999 | 0.6794 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.7722** | **0.7309** | **0.6657** |


### Train: `kodf` → Test: `celeb_df_v2`

| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.5579 | 0.4719 | 1.2605 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.5946** | **0.5400** | **1.0078** |

### FaceForensics++

### Train: `celeb_df_v2` → Test: `ff++`

| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_vit_b0** | 20 | conf | 0 | 0.5241 | 0.6860 | 1.1241 |
| **ms_eff_vit_b5** | 20 | conf | 0 | 0.5533 | 0.7299 | 0.9987 |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.5492 | 0.7301 | 1.0556 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.5825** | **0.7307** | **0.8897** |


### Train: `kodf` → Test: `ff++`

| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | **0.4875** | 0.5341 | 1.6178 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | 0.4525 | **0.5902** | **1.5321** |

### KoDF

### Train: `celeb_df_v2` → Test: `kodf`

| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | **0.4935** | **0.7258** | **1.3459** |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | 0.4832 | 0.7160 | 1.4897 |


### Train: `ff++` → Test: `kodf`

| Model | Frames | Agg Mode | TTA (H-Flip) | Test@Accuracy | Test@AUC | Test@Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ms_eff_gcvit_b0** | 20 | conf | 0 | 0.7544 | 0.862 | 0.8829 |
| **ms_eff_gcvit_b5** | 20 | conf | 0 | **0.7695** | **0.8821** | **0.7635** |

# Inference Latency Comparison

Latency is measured with batch size 1 on a single Tesla T4 and video (num_frames=20).

| Model | Mode | Params (M) | FLOPs (G) | Latency (s) | Ratio |
| ----- | ---- | ---------- | --------- | -------- | ----- |
| **ms_eff_gcvit_b0** | Fast Mode | 8.70 | 0.85 | 0.051 | 1.00× |
| **ms_eff_gcvit_b5** | Pro Mode | 50.34 | 13.43  | 0.479 | 9.41× |


| Model | Params (M) | FLOPs (G) | Latency (s) | Ratio |
| ----- | ---------- | --------- | ----------- | ----- |
| **ms_eff_gcvit_b0 (Ours)** | **8.7** | **0.85** | **0.051** | **1.0×** |
| sia (ECCV 2022) | 17.57 | 2.05 | 0.094 | 1.8× |
| core (CVPR 2022) | 20.81 | 8.40 | 0.153 | 3.0× |
| iid (CVPR 2023) | 65.5 | 12.7 | 0.181 | 3.5× |
| effort (ICML 2025) | 303.4 | 155.9 | 1.041 | 20.4× |

