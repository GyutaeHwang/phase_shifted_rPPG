from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchinfo import summary
from copy import deepcopy
from Dataloader import Overlap_dataset
from datetime import datetime
from random import shuffle
from utils import *
# from torch.utils.tensorboard import SummaryWriter
from scipy import stats
from Shifted_rPPG_extraction import Shifted_rPPG_extraction
from PV_MSELoss import PVLoss
from sklearn.preprocessing import minmax_scale
from matplotlib import gridspec

import torchaudio.transforms as T
import torch.backends.cudnn
import os
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import natsort
import pandas as pd
import gc
from numba import cuda

# Fix random seed for reproducability
seed_number = 0
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed_number)
torch.cuda.manual_seed(seed_number)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_number)

start_time = datetime.now()
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 if os.name == 'nt' else 4

    WINDOW_SIZE = 150
    BATCH_SIZE = 8
    LEARNING_RATE1 = 1e-3
    EPOCH_model1 = 50
    FOLD = 5
    DECAY = 0.99
    phases = ['train', 'test']

    fold_save_path = './record/record_stage1'
    if not os.path.exists(fold_save_path):
        os.mkdir(fold_save_path)

    data_path = "./Preprocessed_MTCNN_size128_scale1.6/"
    subjects_list = []
    data_path_list = []
    for (root, directories, files) in os.walk(data_path):
        for d in directories:
            if 'T' in d:
                subject = d.split('T')[0]
                if subject not in subjects_list:
                    subjects_list.append(subject)

        for f in files:
            if 'data.mat' in f:
                f_path = os.path.join(root, f)
                data_path_list.append(str(f_path))

    data_path_list = natsort.natsorted(data_path_list)

    k_fold_train_subjects_list = []
    k_fold_test_subjects_list = []
    cv = KFold(n_splits=FOLD, shuffle=True)
    for k, (train_idx, test_idx) in enumerate(cv.split(subjects_list)):
        k_fold_train_subjects_list.append(np.array(subjects_list)[train_idx].tolist())
        k_fold_test_subjects_list.append(np.array(subjects_list)[test_idx].tolist())

    # set fold number 0 to 4
    k = 0

    train_subjects_list = k_fold_train_subjects_list[k]
    test_subjects_list = k_fold_test_subjects_list[k]
    subjects_dict = {'train': train_subjects_list, 'test': test_subjects_list}

    save_path = os.path.join(fold_save_path, 'fold' + str(k))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print("test list:", test_subjects_list)

    train_dataset = Overlap_dataset(subjects_dict['train'], data_path_list, WINDOW_SIZE, 'aug', DEVICE)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    test_dataset = Overlap_dataset(subjects_dict['test'], data_path_list, WINDOW_SIZE, 'noaug', DEVICE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    datasets = {'train': train_dataset, 'test': test_dataset}
    dataloaders = {'train': train_loader, 'test': test_loader}

    device = cuda.get_current_device()
    device.reset()

    duration = datetime.now() - start_time
    print('{}th fold data prepared, duration : {}'.format(k, duration))

    Model1 = Shifted_rPPG_extraction()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        Model1 = torch.nn.DataParallel(Model1)
    Model1.to(DEVICE)

    L1_distance = nn.L1Loss()
    L2_distance = nn.MSELoss()
    MSE_loss = nn.MSELoss()
    PV_loss = PVLoss()

    optimizer1 = optim.Adam(Model1.parameters(), lr=LEARNING_RATE1)
    scheduler1 = optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda epoch: DECAY ** epoch)

    global best_model_save_path1
    model1_best_test_MAE = math.inf
    model1_train_list = []
    model1_test_list = []
    for epoch in range(EPOCH_model1):
        print('------ Train rPPG model epoch {}/{} ------'.format(epoch, EPOCH_model1 - 1))
        for phase in phases:
            model1_epoch_face_loss = 0.0
            model1_epoch_face_HR_MAE = 0.0
            model1_epoch_finger_loss = 0.0
            model1_epoch_finger_HR_MAE = 0.0

            if phase == 'train':
                Model1.train()
            else:
                Model1.eval()

            for iteration, (t_x, t_y, t_v) in enumerate(dataloaders[phase]):
                t_x = t_x.to(DEVICE)
                t_y = t_y.to(DEVICE)

                optimizer1.zero_grad()
                face_PPG, finger_PPG = Model1(t_x)

                ref_ppg = (t_y - torch.mean(t_y)) / torch.std(t_y)
                Filtered_ref = torch.zeros_like(ref_ppg, dtype=torch.float32)
                ref_amplitude = torch.zeros((ref_ppg.shape[0], 600), dtype=torch.float32)
                ref_HR = torch.zeros((ref_ppg.shape[0]), dtype=torch.float32).to(DEVICE)

                face_PPG = (face_PPG - torch.mean(face_PPG)) / torch.std(face_PPG)
                Filtered_face_PPG = torch.zeros_like(face_PPG, dtype=torch.float32)
                face_PPG_amplitude = torch.zeros((face_PPG.shape[0], 600) , dtype=torch.float32)
                face_PPG_HR = torch.zeros((face_PPG.shape[0]), dtype=torch.float32).to(DEVICE)

                finger_PPG = (finger_PPG - torch.mean(finger_PPG)) / torch.std(finger_PPG)
                Filtered_finger_PPG = torch.zeros_like(finger_PPG, dtype=torch.float32)
                finger_PPG_amplitude = torch.zeros((finger_PPG.shape[0], 600), dtype=torch.float32)
                finger_PPG_HR = torch.zeros((finger_PPG.shape[0]), dtype=torch.float32).to(DEVICE)

                for b in range(ref_ppg.shape[0]):
                    # reference post-processing
                    y_dtd = torch_detrending_filter(ref_ppg[b, :], device=DEVICE, lam=50)
                    Filtered_ref[b] = torch_butter_bandpass_filter(y_dtd, 0.5, 3, 25, order=3, device=DEVICE)
                    ref_HR[b], _, ref_amplitude[b, :], _ = torch_HR_calculation(Filtered_ref[b], 25, DEVICE)
                    # face post-processing
                    x_dtd = torch_detrending_filter(face_PPG[b, :], device=DEVICE, lam=50)
                    Filtered_face_PPG[b] = torch_butter_bandpass_filter(x_dtd, 0.5, 3, 25, order=3, device=DEVICE)
                    face_PPG_HR[b], _, face_PPG_amplitude[b, :], _ = torch_HR_calculation(Filtered_face_PPG[b], 25, DEVICE)
                    # finger post-processing
                    f_dtd = torch_detrending_filter(finger_PPG[b, :], device=DEVICE, lam=50)
                    Filtered_finger_PPG[b] = torch_butter_bandpass_filter(f_dtd, 0.5, 3, 25, order=3, device=DEVICE)
                    finger_PPG_HR[b], _, finger_PPG_amplitude[b, :], _ = torch_HR_calculation(Filtered_finger_PPG[b], 25, DEVICE)

                # face loss function
                model1_face_freq_loss = MSE_loss(ref_amplitude, face_PPG_amplitude)
                model1_face_time_loss = PV_loss(Filtered_ref, Filtered_face_PPG, DEVICE)
                model1_face_HR_loss = MSE_loss(face_PPG_HR, ref_HR)

                model1_face_iter_loss = 0.0001 * model1_face_HR_loss + 100 * model1_face_freq_loss + model1_face_time_loss
                model1_face_HR_MAE = L1_distance(face_PPG_HR, ref_HR)

                # finger loss function
                model1_finger_freq_loss = MSE_loss(ref_amplitude, finger_PPG_amplitude)
                model1_finger_time_loss = MSE_loss(Filtered_ref, Filtered_finger_PPG) + PV_loss(Filtered_ref, Filtered_finger_PPG, DEVICE)
                model1_finger_HR_loss = MSE_loss(finger_PPG_HR, ref_HR)

                model1_finger_iter_loss = 0.0001 * model1_finger_HR_loss + 100 * model1_finger_freq_loss + model1_finger_time_loss
                model1_finger_HR_MAE = L1_distance(finger_PPG_HR, ref_HR)

                if phase == 'train':
                    model1_face_iter_loss.backward(retain_graph=True)
                    model1_finger_iter_loss.backward()
                    optimizer1.step()

                model1_epoch_face_HR_MAE += model1_face_HR_MAE.item() / len(dataloaders[phase])
                model1_epoch_face_loss += model1_face_iter_loss.item() / len(dataloaders[phase])

                model1_epoch_finger_HR_MAE += model1_finger_HR_MAE.item() / len(dataloaders[phase])
                model1_epoch_finger_loss += model1_finger_iter_loss.item() / len(dataloaders[phase])

            model1_epoch_HR_MAE = (model1_epoch_face_HR_MAE + model1_epoch_finger_HR_MAE)/2
            model1_epoch_loss = (model1_epoch_face_loss + model1_epoch_finger_loss)/2

            print('{}th model1, {}, {} | loss: {:.4f}, {:.4f} | MAE: {:.4f}, {:.4f}'
                  .format(k, epoch, phase, model1_epoch_face_loss, model1_epoch_finger_loss, model1_epoch_face_HR_MAE, model1_epoch_finger_HR_MAE))

            if phase == 'train':
                model1_train_list.append(
                    [k, epoch, phase, model1_epoch_face_loss, model1_epoch_finger_loss, model1_epoch_face_HR_MAE, model1_epoch_finger_HR_MAE])
                performance = pd.DataFrame(model1_train_list,
                                           columns=['fold', 'epoch', 'phase', 'face_loss', 'finger_loss', 'face_HR_MAE', 'finger_HR_MAE'])
                record_file_path = os.path.join(save_path, str(k) + 'th fold model1 train results.xlsx')
                performance.to_excel(record_file_path)
            if phase == 'test':
                model1_test_list.append(
                    [k, epoch, phase, model1_epoch_face_loss, model1_epoch_finger_loss, model1_epoch_face_HR_MAE, model1_epoch_finger_HR_MAE])
                performance = pd.DataFrame(model1_test_list,
                                           columns=['fold', 'epoch', 'phase', 'face_loss', 'finger_loss', 'face_HR_MAE', 'finger_HR_MAE'])
                record_file_path = os.path.join(save_path, str(k) + 'th fold model1 test results.xlsx')
                performance.to_excel(record_file_path)

            weight_path = os.path.join(save_path, 'weights')
            if not os.path.exists(weight_path):
                os.mkdir(weight_path)

            if phase == 'test':
                if model1_epoch_HR_MAE < model1_best_test_MAE:
                    best_model_state_dict1 = deepcopy(Model1.state_dict())
                    model1_best_test_MAE = model1_epoch_HR_MAE
                    best_model_save_path1 = os.path.join(weight_path, 'best_model1_state_dict_LOSS_{:.6f}_MAE_{:.6f}_EPOCH_{:03d}_LR_{}_BATCH_{:03d}_DECAY_{}.pth'
                                                         .format(model1_epoch_loss, model1_epoch_HR_MAE, epoch, LEARNING_RATE1, BATCH_SIZE, DECAY))
                    torch.save(best_model_state_dict1, best_model_save_path1)
        scheduler1.step()

    duration = datetime.now() - start_time
    print('{}th fold train finished, duration : {}'.format(k, duration))

    Model1.apply(weight_reset)
    Model1.load_state_dict(torch.load(best_model_save_path1))
    Model1.to(DEVICE)
    Model1.eval()

    inference_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    model1_total_face_HR_MAE = 0.0
    model1_total_finger_HR_MAE = 0.0
    model1_total_face_HR_MSE = 0.0
    model1_total_finger_HR_MSE = 0.0

    Facial_HR_list = []
    acral_HR_list = []
    ref_HR_list = []

    for samples, (t_x, t_y, t_v) in enumerate(inference_loader):
        t_x = t_x.to(DEVICE)
        t_y = t_y.to(DEVICE)

        face_PPG, finger_PPG = Model1(t_x)

        ref_ppg = (t_y - torch.mean(t_y)) / torch.std(t_y)
        ref_ppg = ref_ppg.view(ref_ppg.shape[1])
        y_dtd = torch_detrending_filter(ref_ppg, device=DEVICE, lam=50)
        Filtered_ref = torch_butter_bandpass_filter(y_dtd, 0.5, 3, 25, order=3, device=DEVICE)
        ref_HR, ref_frequency, ref_amplitude, ref_peak_frequency = torch_HR_calculation(Filtered_ref, 25, DEVICE)
        ref_ppg_cpu = Filtered_ref.detach().cpu().numpy()

        face_pred_ppg = (face_PPG - torch.mean(face_PPG)) / torch.std(face_PPG)
        face_pred_ppg = face_pred_ppg.view(face_pred_ppg.shape[1])
        x_dtd = torch_detrending_filter(face_pred_ppg, device=DEVICE, lam=50)
        Filtered_face_pred = torch_butter_bandpass_filter(x_dtd, 0.5, 3, 25, order=3, device=DEVICE)
        pred_face_HR, pred_face_frequency, pred_face_amplitude, pred_face_peak_frequency = torch_HR_calculation(Filtered_face_pred, 25, DEVICE)
        pred_face_ppg_cpu = Filtered_face_pred.detach().cpu().numpy()

        finger_pred_ppg = (finger_PPG - torch.mean(finger_PPG)) / torch.std(finger_PPG)
        finger_pred_ppg = finger_pred_ppg.view(finger_pred_ppg.shape[1])
        f_dtd = torch_detrending_filter(finger_pred_ppg, device=DEVICE, lam=50)
        Filtered_finger_pred = torch_butter_bandpass_filter(f_dtd, 0.5, 3, 25, order=3, device=DEVICE)
        pred_finger_HR, pred_finger_frequency, pred_finger_amplitude, pred_finger_peak_frequency = torch_HR_calculation(Filtered_finger_pred, 25, DEVICE)
        pred_finger_ppg_cpu = Filtered_finger_pred.detach().cpu().numpy()

        model1_face_HR_MAE = L1_distance(pred_face_HR, ref_HR)
        model1_finger_HR_MAE = L1_distance(pred_finger_HR, ref_HR)
        model1_total_face_HR_MAE += model1_face_HR_MAE.item() / len(inference_loader)
        model1_total_finger_HR_MAE += model1_finger_HR_MAE.item() / len(inference_loader)

        model1_face_HR_MSE = L2_distance(pred_face_HR, ref_HR)
        model1_finger_HR_MSE = L2_distance(pred_finger_HR, ref_HR)
        model1_total_face_HR_MSE += model1_face_HR_MSE.item() / len(inference_loader)
        model1_total_finger_HR_MSE += model1_finger_HR_MSE.item() / len(inference_loader)

        Facial_HR_list.append(pred_face_HR.item())
        acral_HR_list.append(pred_finger_HR.item())
        ref_HR_list.append(ref_HR.item())

    Facial_HR_array = np.array(Facial_HR_list)
    acral_HR_array = np.array(acral_HR_list)
    ref_HR_array = np.array(ref_HR_list)

    Facial_r = stats.pearsonr(ref_HR_array, Facial_HR_array)
    Acral_r = stats.pearsonr(ref_HR_array, acral_HR_array)

    print('k: {} | MAE: {:.4f}, {:.4f} | RMSE: {:.4f}, {:.4f} | r: {:.4f}, {:.4f}'.format(k,
                                                                                              model1_total_face_HR_MAE,
                                                                                              model1_total_finger_HR_MAE,
                                                                                              np.sqrt(
                                                                                                  model1_total_face_HR_MSE),
                                                                                              np.sqrt(
                                                                                                  model1_total_finger_HR_MSE),
                                                                                              Facial_r[0], Acral_r[0]))

duration = datetime.now() - start_time
print('EXPERIMENT OVER, duration : {}'.format(duration))

