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


## Citation

```
@inproceedings{Celeb_DF_cvpr20,
   author = {Yuezun Li, Xin Yang, Pu Sun, Honggang Qi and Siwei Lyu},
   title = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
   booktitle= {IEEE Conference on Computer Vision and Patten Recognition (CVPR)},
   year = {2020}
}
```