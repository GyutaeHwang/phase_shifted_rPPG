from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from scipy import sparse
from scipy.stats import skew
import torchaudio.functional
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def torch_HR_calculation(signal, fps, device):
    # perform FFT for power spectrum analysis
    window = torch.hamming_window(signal.shape[0]).to(device)
    fft = torch.fft.fft(signal*window, n=(signal.shape[0]*40)) / signal.shape[0]
    amplitude = torch.abs(fft)
    frequency = torch.fft.fftfreq(len(fft), 1 / fps, device=device, requires_grad=True)

    filtered_idx = torch.where((0.5 <= frequency) & (frequency <= 3))
    filtered_frequency = frequency[filtered_idx]
    fitered_amplitude = amplitude[filtered_idx]

    peak_frequency = filtered_frequency[torch.argmax(fitered_amplitude)]

    return peak_frequency * 60, filtered_frequency, fitered_amplitude, peak_frequency


def HR_calculation(signal, fps):
    # perform FFT for power spectrum analysis
    window = np.hamming(signal.shape[0])
    fft = np.fft.fft(signal * window, n=(signal.shape[0] * 40)) / signal.shape[0]
    amplitude = abs(fft)
    frequency = np.fft.fftfreq(len(fft), 1 / fps)

    # filtering HR, 30~180 BPM
    filtered_idx = np.where((0.5 <= frequency) & (frequency <= 3))
    filtered_frequency = frequency[filtered_idx]
    filtered_amplitude = amplitude[filtered_idx]

    # Find peak frequency value
    peak_frequency = filtered_frequency[np.argmax(filtered_amplitude)]

    return peak_frequency * 60, filtered_frequency, filtered_amplitude, peak_frequency


def torch_butter_bandpass_filter(signal, lowcut, highcut, fs, order, device):
    '''

    :param signal: (N, ) shape // (N, 1) 안됨!
    '''
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    signal_clone = signal.clone() # backward의 inplace error 때문에 clone()
    b_tensor = torch.FloatTensor(b).to(device)
    a_tensor = torch.FloatTensor(a).to(device)
    y = torchaudio.functional.filtfilt(signal_clone, a_tensor, b_tensor, clamp=False)
    return y


def butter_bandpass_filter(signal, lowcut, highcut, fs, order):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = filtfilt(b, a, signal)
    '''
    filtfilt → zero phase filter for offline filtering (not for online)
    '''
    return y


def torch_detrending_filter(signal, device, lam=50):
    T = len(signal)
    I = sparse.eye(T).toarray()
    D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T - 2, T)).toarray()

    # torch Tensor
    sr = signal

    # numpy
    op = np.multiply(np.transpose(D2),lam**2)
    op2 = np.matmul(op, D2) + I

    # torch Tensor
    op2 = torch.FloatTensor(op2).to(device)

    op3 = torch.linalg.solve(op2, sr) # np.matmul(np.linalg.inv(op2), sr), ref: An Advanced Detrending Method With Application to HRV Analysis (2002, IEEE TOBE)

    detrended_signal = torch.matmul(torch.FloatTensor(I).to(device), sr) - op3

    return detrended_signal


def metric(pred_list, ref_list, value, save_path):
    MAE = mean_absolute_error(pred_list, ref_list)
    MSE = mean_squared_error(pred_list, ref_list)
    RMSE = np.sqrt(MSE)
    r = pearsonr(pred_list, ref_list)[0]

    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(pred_list, ref_list, c='green', alpha=0.5, s=0.8, label=value)
    plt.title("{}_MAE: {:.3f}, RMSE: {:.3f}, r: {:.3f}".format(value, MAE, RMSE, r))
    plt.legend()
    plt.xlabel('Prediction')
    plt.ylabel('Reference')
    plt.savefig(os.path.join(save_path, value + '_scatter.png'))
    plt.cla()

    return MAE, RMSE, r


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

