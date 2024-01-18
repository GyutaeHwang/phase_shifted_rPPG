# Phase-shifted remote photoplethysmography for estimating heart rate and blood pressure from facial video

Pytorch implementation of phase-shifted rPPG for estimating heart rate and blood pressure from facial video.

-----------
## Overview

Cardiovascular diseases, such as hypertension, arrhythmias, and stroke, represent a primary contributor to the worldwide increase in mortality rates. Monitoring heart rate and blood pressure is crucial for preventing cardiovascular diseases. Therefore, recent research has been conducted on deep learning-based estimation of physiological information from camera. In this repository, we implement a deep learning framework for estimating rPPG and heart rate from different body regions based on video. Moreover, we utilize the phase differences in the pulse cycle to estimate blood pressure. In detail, we designed a frame interpolation-based augmentation technique for heart rate data and developed DRP-Net for estimating phase-shifted rPPG. To estimate blood pressure, we designed BBP-Net, which takes phase-shifted rPPG as input, and employed a scaling sigmoid function to bound the blood pressure within a reasonable range. To demonstrate the effectiveness of the proposed method, experiments were conducted on publicly available datasets, MMSE-HR and V4V datasets.

-----------
## Demo video

<p align="center">
<img src = "https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/8451646c-6c52-4826-92c4-ee79ea82ea33.gif"/>
</p>

You can watch the full demo video in [HERE](https://youtu.be/t-BFKd023L4).

## DRP-Net architecture

<p align="center">
<img src = "https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/dc22ffb8-f731-4de1-b3a8-d4a72408bf28.png" width="60%" height="60%"/>
</p>

## BBP-Net architecture

<p align="center">
<img src = "https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/7f7692a4-a249-4932-9a75-47e7c2e1faae.png"  width="60%" height="60%"/>
</p>

-----------
## Installation

### Requirements
- Ubuntu 20.04
- CUDA 11.3
- Python v3.9.11
- Pytorch v1.12.1

### Environment setup
- Create the conda environment
```bash
conda create -n phase_shifted_rPPG python=3.9
conda activate phase_shifted_rPPG
pip install -r requirements.txt
```

### Dataset download
You can download MMSE-HR database from [HERE](https://binghamton.technologypublisher.com/tech/MMSE-HR_dataset_(Multimodal_Spontaneous_Expression-Heart_Rate_dataset)).

### Preprocessing
- Requires entering a path to the dataset
```bash
python3 Preprocessing.py
```

### Train DRP-Net
```bash
python3 main_stage1.py
```

### Train BBP-Net
- Requires the pretrained parameter of DRP-Net
```bash
python3 main_stage2.py
```
-----------

## Acknowledgement
**Datasets for the experiments: MMSE-HR, V4V**
<br/>Zhang, Zheng, et al. "Multimodal spontaneous emotion corpus for human behavior analysis." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
<br/>Revanur, Ambareesh, et al. "The first vision for vitals (v4v) challenge for non-contact video-based physiological estimation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

**Face detection method: MTCNN**
<br/>Xiang, Jia, and Gengming Zhu. "Joint face detection and facial expression recognition with MTCNN." 2017 4th international conference on information science and control engineering (ICISCE). IEEE, 2017.

**Frame interpolation method: FILM**
<br/>Reda, Fitsum, et al. "Film: Frame interpolation for large motion." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.

-----------

@article{hwang2024phase,
  <br/>title={Phase-shifted remote photoplethysmography for estimating heart rate and blood pressure from facial video},
  <br/>author={Hwang, Gyutae and Lee, Sang Jun},
  <br/>journal={arXiv preprint arXiv:2401.04560},
  <br/>year={2024}
<br/>}
