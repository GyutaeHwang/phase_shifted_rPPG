# Phase shifted rPPG for estimating heart rate and blood pressure from facial video

-----------

## Demo video
[![Video Label](http://img.youtube.com/vi/t-BFKd023L4/0.jpg)](https://youtu.be/t-BFKd023L4)

## DRP-Net
![3_DRP-Net_배경](https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/dc22ffb8-f731-4de1-b3a8-d4a72408bf28)

## BBP-Net
![4_BBP-Net_배경](https://github.com/GyutaeHwang/phase_shifted_rPPG/assets/93236013/7f7692a4-a249-4932-9a75-47e7c2e1faae)

-----------
## Dataset download
You can download MMSE-HR database from [HERE](https://binghamton.technologypublisher.com/tech/MMSE-HR_dataset_(Multimodal_Spontaneous_Expression-Heart_Rate_dataset))

## Preprocessing
'''python
python3 Preprocessing.py
'''

## Train DRP-Net
'''python
python3 main_stage1.py
'''

## Train BBP-Net
'''python
python3 main_stage2.py
'''
-----------

## Acknowledgement
MMSE-HR database
V4V database
MTCNN
FILM

@article{hwang2024phase,
  title={Phase-shifted remote photoplethysmography for estimating heart rate and blood pressure from facial video},
  author={Hwang, Gyutae and Lee, Sang Jun},
  journal={arXiv preprint arXiv:2401.04560},
  year={2024}
}
