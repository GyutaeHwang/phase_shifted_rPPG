import torch
import torch.nn as nn
import numpy as np
from scipy.signal import argrelextrema
from utils import *


def PV_detection(signal, diff):
    peaks = []
    valleys = []
    SBP = []
    DBP = []

    for i in range(diff.shape[0] - 1):
        if diff[i] > 0:
            if diff[i + 1] < 0:
                peaks.append(signal[i + 1])

    for j in range(diff.shape[0] - 1):
        if diff[j] < 0:
            if diff[j + 1] > 0:
                valleys.append(signal[j + 1])

    for p in range(len(peaks)):
        if peaks[p] >= torch.mean(torch.stack(peaks)):
            SBP.append(peaks[p])

    for v in range(len(valleys)):
        if valleys[v] <= torch.mean(torch.stack(valleys)):
            DBP.append(valleys[v])

    return torch.mean(torch.stack(SBP)), torch.mean(torch.stack(DBP))


class PVLoss(nn.Module):
    def __init__(self):
        super(PVLoss, self).__init__()
        self.MSE_loss = nn.MSELoss()

    def forward(self, preds, labels, DEVICE):

        pred_values = torch.zeros((preds.shape[0], 2), dtype=torch.float32).to(DEVICE)
        label_values = torch.zeros_like(pred_values, dtype=torch.float32).to(DEVICE)
        pred_diffs = torch.diff(preds, dim=1)
        label_diffs = torch.diff(labels, dim=1)

        for i in range(labels.shape[0]):
            label_values[i, 0], label_values[i, 1] = PV_detection(labels[i], label_diffs[i])
            pred_values[i, 0], pred_values[i, 1] = PV_detection(preds[i], pred_diffs[i])

        loss = self.MSE_loss(pred_values, label_values)
        return loss

