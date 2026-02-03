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


## Citation

```
@inproceedings{roessler2019faceforensicspp,
	author = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
	title = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
	booktitle= {International Conference on Computer Vision (ICCV)},
	year = {2019}
}
```