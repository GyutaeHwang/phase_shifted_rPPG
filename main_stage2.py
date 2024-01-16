from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from copy import deepcopy
from numba import cuda
from datetime import datetime
from scipy import stats
from Shifted_rPPG_extraction import Shifted_rPPG_extraction
from BP_estimation import BP_estimation
from Dataloader import Overlap_dataset
from utils import *
from matplotlib import gridspec
import statsmodels.api as sm

import matplotlib.image as img
import scipy.signal
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
import cv2

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
    TRAIN_STRIDE = 100
    TEST_STRIDE = WINDOW_SIZE
    EPOCH_model2 = 200
    BATCH_SIZE = 8
    LEARNING_RATE2 = 1e-3
    FOLD = 5
    DECAY = 0.99
    phases = ['train', 'test']

    fold_save_path = './record/record_stage2'
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

    k=0
    train_subjects_list = k_fold_train_subjects_list[k]
    test_subjects_list = k_fold_test_subjects_list[k]
    subjects_dict = {'train': train_subjects_list, 'test': test_subjects_list}

    save_path = os.path.join(fold_save_path, 'fold' + str(k))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("test list:", test_subjects_list)

    train_dataset = Overlap_dataset(subjects_dict['train'], data_path_list, WINDOW_SIZE, 'noaug', DEVICE)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, drop_last=True)
    test_dataset = Overlap_dataset(subjects_dict['test'], data_path_list, WINDOW_SIZE, 'noaug', DEVICE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    datasets = {'train': train_dataset, 'test': test_dataset}
    dataloaders = {'train': train_loader, 'test': test_loader}

    device = cuda.get_current_device()
    device.reset()

    duration = datetime.now() - start_time
    print('{}th fold data prepared, duration : {}'.format(k, duration))

    Model1 = Shifted_rPPG_extraction()
    Model2 = BP_estimation()

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        Model1 = torch.nn.DataParallel(Model1)
        Model2 = torch.nn.DataParallel(Model2)
    Model1.to(DEVICE)
    Model2.to(DEVICE)

    Model1_save_path_list = os.listdir('./record/record_23072601_6s/fold' + str(k) + '/weights')
    Model1_save_path = os.path.join('./record/record_23072601_6s/fold' + str(k) + '/weights', Model1_save_path_list[0])

    Model1.apply(weight_reset)
    Model1.load_state_dict(torch.load(Model1_save_path))
    Model1.to(DEVICE)
    Model1.eval()

    L1_distance = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    Huber_loss = nn.HuberLoss()

    optimizer2 = optim.Adam(Model2.parameters(), lr=LEARNING_RATE2)
    scheduler2 = optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=lambda epoch: DECAY ** epoch)

    global best_model_save_path2
    model2_best_test_MAE = math.inf
    model1_train_list = []
    model1_test_list = []
    for epoch in range(EPOCH_model2):
        print('------ Train BP model epoch {}/{} ------'.format(epoch, EPOCH_model2 - 1))
        for phase in phases:
            model2_epoch_SBP_loss = 0.0
            model2_epoch_DBP_loss = 0.0
            model2_epoch_SBP_MAE = 0.0
            model2_epoch_DBP_MAE = 0.0
            model2_epoch_signal_loss = 0.0

            if phase == 'train':
                Model1.eval()
                Model2.train()
            else:
                Model1.eval()
                Model2.eval()

            for iteration, (t_x, t_y, t_v) in enumerate(dataloaders[phase]):
                t_x = t_x.to(DEVICE)
                t_y = t_y.to(DEVICE)
                t_v = t_v.to(DEVICE)

                face_PPG, finger_PPG = Model1(t_x)

                face_PPG = scipy.signal.resample(face_PPG.detach().cpu().numpy(), WINDOW_SIZE * 4, axis=1)
                finger_PPG = scipy.signal.resample(finger_PPG.detach().cpu().numpy(), WINDOW_SIZE * 4, axis=1)
                ABP = scipy.signal.resample(t_y.detach().cpu().numpy(), WINDOW_SIZE * 4, axis=1)

                face_PPG = torch.from_numpy(face_PPG).type(torch.FloatTensor).to(DEVICE)
                finger_PPG = torch.from_numpy(finger_PPG).type(torch.FloatTensor).to(DEVICE)
                ABP = torch.from_numpy(ABP).type(torch.FloatTensor).to(DEVICE)

                face_PPG = (face_PPG - torch.mean(face_PPG)) / torch.std(face_PPG)
                finger_PPG = (finger_PPG - torch.mean(finger_PPG)) / torch.std(finger_PPG)

                optimizer2.zero_grad()

                SBP, DBP = Model2(face_PPG, finger_PPG, DEVICE)

                scaled_finger_PPG = torch.zeros_like(ABP, dtype=torch.float32)
                for b in range(ABP.shape[0]):
                    std = (finger_PPG[b] - torch.min(finger_PPG[b])) / (torch.max(finger_PPG[b]) - torch.min(finger_PPG[b]))
                    scaled_finger_PPG[b] = std * (SBP[b] - DBP[b]) + DBP[b]

                model2_SBP_iter_loss = Huber_loss(SBP, t_v[:, 0])
                model2_DBP_iter_loss = Huber_loss(DBP, t_v[:, 1])
                model2_signal_iter_loss = MSE_loss(scaled_finger_PPG, ABP)

                model2_SBP_MAE = L1_distance(SBP, t_v[:, 0])
                model2_DBP_MAE = L1_distance(DBP, t_v[:, 1])

                if phase == 'train':
                    model2_SBP_iter_loss.backward(retain_graph=True)
                    model2_DBP_iter_loss.backward(retain_graph=True)
                    model2_signal_iter_loss.backward()
                    optimizer2.step()

                model2_epoch_SBP_MAE += model2_SBP_MAE.item() / len(dataloaders[phase])
                model2_epoch_SBP_loss += model2_SBP_iter_loss.item() / len(dataloaders[phase])

                model2_epoch_DBP_MAE += model2_DBP_MAE.item() / len(dataloaders[phase])
                model2_epoch_DBP_loss += model2_DBP_iter_loss.item() / len(dataloaders[phase])

                model2_epoch_signal_loss += model2_signal_iter_loss.item() / len(dataloaders[phase])

            model2_epoch_BP_MAE = (model2_epoch_SBP_MAE + model2_epoch_DBP_MAE) / 2
            model2_epoch_loss = (model2_epoch_SBP_loss + model2_epoch_DBP_loss + model2_epoch_signal_loss) / 2

            print('{}th model1, {}, {} | loss: {:.4f}, {:.4f} | MAE: {:.4f}, {:.4f}'
                  .format(k, epoch, phase, model2_epoch_SBP_loss, model2_epoch_DBP_loss, model2_epoch_SBP_MAE, model2_epoch_DBP_MAE))

            if phase == 'train':
                model1_train_list.append(
                    [k, epoch, phase, model2_epoch_SBP_loss, model2_epoch_DBP_loss, model2_epoch_SBP_MAE, model2_epoch_DBP_MAE])
                performance = pd.DataFrame(model1_train_list,
                                           columns=['fold', 'epoch', 'phase', 'SBP_loss', 'DBP_loss', 'SBP_MAE', 'DBP_MAE'])
                record_file_path = os.path.join(save_path, str(k) + 'th model1 train results.xlsx')
                performance.to_excel(record_file_path)
            if phase == 'test':
                model1_test_list.append(
                    [k, epoch, phase, model2_epoch_SBP_loss, model2_epoch_DBP_loss, model2_epoch_SBP_MAE, model2_epoch_DBP_MAE])
                performance = pd.DataFrame(model1_test_list,
                                           columns=['fold', 'epoch', 'phase', 'SBP_loss', 'DBP_loss', 'SBP_MAE', 'DBP_MAE'])
                record_file_path = os.path.join(save_path, str(k) + 'th model1 test results.xlsx')
                performance.to_excel(record_file_path)

            weight_path = os.path.join(save_path, 'weights')
            if not os.path.exists(weight_path):
                os.mkdir(weight_path)

            if phase == 'test':
                if model2_epoch_BP_MAE < model2_best_test_MAE:
                    best_model_state_dict2 = deepcopy(Model2.state_dict())
                    model2_best_test_MAE = model2_epoch_BP_MAE
                    best_model_save_path2 = os.path.join(weight_path,
                                                         'best_model2_state_dict_LOSS_{:.6f}_MAE_{:.6f}_EPOCH_{:03d}_LR_{}_BATCH_{:03d}_DECAY_{}.pth'
                                                         .format(model2_epoch_loss, model2_epoch_BP_MAE, epoch,
                                                                 LEARNING_RATE2, BATCH_SIZE, DECAY))
                    torch.save(best_model_state_dict2, best_model_save_path2)
        scheduler2.step()

        duration = datetime.now() - start_time
        print('{}th fold train finished, duration : {}'.format(k, duration))

    Model2.apply(weight_reset)
    Model2.load_state_dict(torch.load(best_model_save_path2))
    Model2.to(DEVICE)
    Model2.eval()

    inference_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    L2_distance = nn.MSELoss()

    Facial_HR_list = []
    acral_HR_list = []
    ref_HR_list = []

    facial_rPPG_list = []
    acral_rPPG_list = []
    ref_ppg_list = []

    model2_epoch_SBP_MAE = 0.0
    model2_epoch_DBP_MAE = 0.0
    model2_epoch_SBP_MSE = 0.0
    model2_epoch_DBP_MSE = 0.0

    pred_SBP_list = []
    pred_DBP_list = []
    ref_SBP_list = []
    ref_DBP_list = []
    for samples, (t_x, t_y, t_v) in enumerate(inference_loader):
        t_x = t_x.to(DEVICE)
        t_y = t_y.to(DEVICE)
        t_v = t_v.to(DEVICE)

        face_PPG, finger_PPG = Model1(t_x)

        face_PPG = scipy.signal.resample(face_PPG.detach().cpu().numpy(), WINDOW_SIZE * 4, axis=1)
        finger_PPG = scipy.signal.resample(finger_PPG.detach().cpu().numpy(), WINDOW_SIZE * 4, axis=1)

        face_PPG = torch.from_numpy(face_PPG).type(torch.FloatTensor).to(DEVICE)
        finger_PPG = torch.from_numpy(finger_PPG).type(torch.FloatTensor).to(DEVICE)

        face_PPG = (face_PPG - torch.mean(face_PPG)) / torch.std(face_PPG)
        finger_PPG = (finger_PPG - torch.mean(finger_PPG)) / torch.std(finger_PPG)

        SBP, DBP = Model2(face_PPG, finger_PPG, DEVICE)

        model2_SBP_MAE = L1_distance(SBP, t_v[:, 0])
        model2_DBP_MAE = L1_distance(DBP, t_v[:, 1])
        model2_SBP_MSE = MSE_loss(SBP, t_v[:, 0])
        model2_DBP_MSE = MSE_loss(DBP, t_v[:, 1])

        model2_epoch_SBP_MAE += model2_SBP_MAE.item() / len(inference_loader)
        model2_epoch_DBP_MAE += model2_DBP_MAE.item() / len(inference_loader)
        model2_epoch_SBP_MSE += model2_SBP_MSE.item() / len(inference_loader)
        model2_epoch_DBP_MSE += model2_DBP_MSE.item() / len(inference_loader)

    print(
        'k: {} | BP MAE: {:.4f}, {:.4f} | RMSE: {:.4f}, {:.4f} | r: {:.4f}, {:.4f}'.format(k, model2_epoch_SBP_MAE,
                                                                                            model2_epoch_DBP_MAE,
                                                                                            np.sqrt(
                                                                                                model2_epoch_SBP_MSE),
                                                                                            np.sqrt(
                                                                                                model2_epoch_DBP_MSE),
                                                                                            SBP_r[0], DBP_r[0]))


duration = datetime.now() - start_time
print('EXPERIMENT OVER, duration : {}'.format(duration))
