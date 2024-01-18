# Phase shifted rPPG for estimating heart rate and blood pressure from facial video

-----------

## Demo video
![phase_shifted_rPPG-ezgif com-video-to-gif-converter](https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/8451646c-6c52-4826-92c4-ee79ea82ea33)

You can watch the original demo video in [HERE](https://youtu.be/t-BFKd023L4)

## DRP-Net

<img src = "https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/dc22ffb8-f731-4de1-b3a8-d4a72408bf28.png" width="70%" height="70%"/>

## BBP-Net

<img src = "https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/7f7692a4-a249-4932-9a75-47e7c2e1faae.png"  width="70%" height="70%"/>

-----------
## Train models
### 1. Dataset download
You can download MMSE-HR database from [HERE](https://binghamton.technologypublisher.com/tech/MMSE-HR_dataset_(Multimodal_Spontaneous_Expression-Heart_Rate_dataset)).

### 2. Preprocessing
```bash
python3 Preprocessing.py
```

### 3. Train DRP-Net
```bash
python3 main_stage1.py
```

### 4. Train BBP-Net
```bash
python3 main_stage2.py
```
-----------

## Acknowledgement
**Datasets for the experiments: MMSE-JR, V4V**
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
