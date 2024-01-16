import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from scipy.signal import argrelextrema
import scipy.signal
import natsort
from datetime import datetime
import gc
from numba import cuda
import interpolator as interpolator_lib
from utils import *
import random


def Value_extraction(ref, DEVICE):
    ref_ppg = torch.from_numpy(ref).type(torch.FloatTensor)
    ref_ppg = (ref_ppg - torch.mean(ref_ppg)) / torch.std(ref_ppg)
    y_dtd = torch_detrending_filter(ref_ppg.to(DEVICE), device=DEVICE, lam=50)
    Filtered_ref = torch_butter_bandpass_filter(y_dtd, 0.5, 3, 25, order=3, device=DEVICE)

    ref_HR,_,_,_ = torch_HR_calculation(Filtered_ref, 25, DEVICE)
    ref = np.squeeze(ref)
    ref_peaks = argrelextrema(ref, np.greater, order=10)
    ref_vallys = argrelextrema(ref, np.less, order=10)
    ref_SBP = np.mean(ref[ref_peaks])
    ref_DBP = np.mean(ref[ref_vallys])

    return ref_SBP, ref_DBP, ref_HR.item()


def FILM(half_window, interpolator):
    interpolated_video = np.zeros((int((half_window.shape[0]-1) * 2), half_window.shape[1], half_window.shape[2], half_window.shape[3]),
                                  dtype=np.float32)

    for frame in range((half_window.shape[0])-1):
        image_batch_1 = np.expand_dims(half_window[frame, :, :, :], axis=0)
        interpolated_video[frame * 2, :, :, :] = half_window[frame, :, :, :]
        image_batch_2 = np.expand_dims(half_window[frame + 1, :, :, :], axis=0)
        batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

        interpolated_video[(frame * 2)+1, :, :, :] = interpolator(image_batch_1, image_batch_2, batch_dt)[0]

    return interpolated_video


class Overlap_dataset(Dataset):
    def __init__(self, subjects, data_path, WINDOW_SIZE, mode, DEVICE):
        super(Overlap_dataset, self).__init__()

        self.WINDOW_SIZE = WINDOW_SIZE
        self.STRIDE = self.WINDOW_SIZE
        self.DEVICE = 'cpu'

        interpolator = interpolator_lib.Interpolator(model_path='pretrained_FILM/saved_model', align=1, block_shape=[1, 1])

        X = []
        y = []
        values = []
        for subject in subjects:
            for path in data_path:
                if subject in path:
                    print(path)
                    data = sio.loadmat(path)

                    name = path.split('/')[6]

                    tmp_x = data['video']
                    tmp_y = np.squeeze(data['ref'])
                    tmp_x_win = []
                    tmp_y_win = []
                    tmp_values_win = []
                    win_start = 0
                    win_end = win_start + self.WINDOW_SIZE

                    while win_end <= len(tmp_x):
                        tmp_value = Value_extraction(tmp_y[win_start:win_end], self.DEVICE)
                        if mode == 'aug':
                            tmp_x_win.append(tmp_x[win_start:win_end])
                            tmp_y_win.append(tmp_y[win_start:win_end])
                            tmp_values_win.append(tmp_value)

                        if mode == 'noaug':
                            if tmp_value[0] <= 80 or 160 <= tmp_value[0]:
                                pass

                            elif tmp_value[1] <= 40 or 100 <= tmp_value[1]:
                                pass

                            else:
                                tmp_x_win.append(tmp_x[win_start:win_end])
                                tmp_y_win.append(tmp_y[win_start:win_end])
                                tmp_values_win.append(tmp_value)


                        if mode == 'aug':
                            # HRDA - temporally upsample
                            if (75 <= tmp_value[2]) & (tmp_value[2] <= 90):
                                half_y_win = tmp_y[win_start:int(win_start + self.WINDOW_SIZE / 2)]
                                interpolated_half_y_win = scipy.signal.resample(half_y_win, self.WINDOW_SIZE)
                                half_tmp_value = Value_extraction(interpolated_half_y_win, self.DEVICE)

                                if half_tmp_value[2] <= tmp_value[2]*3/4:

                                    half_x_win = tmp_x[win_start:int(win_start+(self.WINDOW_SIZE / 2 + 1))]
                                    upsampled = FILM(half_x_win, interpolator)
                                    tmp_x_win.append(upsampled)
                                    tmp_y_win.append(interpolated_half_y_win)
                                    tmp_values_win.append(half_tmp_value)
                                    print("Upsampled", tmp_value[2], half_tmp_value[2])

                            # HRDA - temporally downsample
                            if (70 <= tmp_value[2]) & (tmp_value[2] <= 80):
                                double_x_win = tmp_x[win_start:int(win_start + (self.WINDOW_SIZE * 2))]
                                if self.WINDOW_SIZE * 2 <= double_x_win.shape[0]:
                                    sampled_double_x_win = np.zeros((self.WINDOW_SIZE, double_x_win.shape[1], double_x_win.shape[2], double_x_win.shape[3]),
                                      dtype=np.float32)
                                    for frame in range(self.WINDOW_SIZE):
                                        sampled_double_x_win[frame, :, :, :] = double_x_win[frame*2, :, :, :]
                                    tmp_x_win.append(sampled_double_x_win)

                                    double_y_win = tmp_y[win_start:int(win_start + (self.WINDOW_SIZE * 2)-1)]
                                    sampled_double_y_win = scipy.signal.resample(double_y_win, self.WINDOW_SIZE)
                                    tmp_y_win.append(sampled_double_y_win)
                                    double_tmp_value = Value_extraction(sampled_double_y_win, self.DEVICE)
                                    tmp_values_win.append(double_tmp_value)
                                    print("Downsampled", tmp_value[2], double_tmp_value[2])

                        win_start += self.STRIDE
                        win_end += self.STRIDE

                    if len(tmp_x_win) == 0:
                        pass

                    else:
                        tmp_x_win = np.array(tmp_x_win)
                        tmp_y_win = np.array(tmp_y_win)
                        tmp_values_win = np.array(tmp_values_win)
                        tmp_x_win = torch.from_numpy(tmp_x_win).type(torch.FloatTensor)
                        tmp_y_win = torch.from_numpy(tmp_y_win).type(torch.FloatTensor)
                        tmp_values_win = torch.from_numpy(tmp_values_win).type(torch.FloatTensor)
                        tmp_x_win = tmp_x_win.permute(0, 4, 1, 2, 3)

                        X.append(tmp_x_win)
                        y.append(tmp_y_win)
                        values.append(tmp_values_win)

        self.X = torch.cat(X)
        print(self.X.shape)
        del X
        self.y = torch.cat(y)
        del y
        self.values = torch.cat(values)
        del values

    def __getitem__(self, index):
        video_clip = self.X[index, :, :, :]
        signal_clip = self.y[index, :]
        value_clip = self.values[index, :]

        return video_clip, signal_clip, value_clip

    def __len__(self):
        return self.X.shape[0]


