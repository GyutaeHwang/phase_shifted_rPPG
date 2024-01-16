import os
import cv2
import numpy as np
import os
from tqdm import tqdm
import scipy.signal
from datetime import datetime
import natsort
import torch
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import scipy.io as sio

device = torch.device("cuda:0")

def Video_cropping(frames, device, box_scale_factor):
    mtcnn = MTCNN(device=device)
    first_frame = frames[0]
    H, W = first_frame.shape[:2]
    box, _ = mtcnn.detect(first_frame)

    n = 0
    while len(box) != 1:
        next_frame = frames[n+1]
        box, _ = mtcnn.detect(next_frame)
        n += 1

    box = np.array(box, dtype=int).squeeze()
    x_1, y_1 = box[:2]
    x_2, y_2 = box[2:]

    add_x = (x_2 - x_1) * (box_scale_factor - 1)
    add_y = (y_2 - y_1) * (box_scale_factor - 1)
    scaled_x_1 = x_1 - int(add_x / 2)
    scaled_x_2 = x_2 + int(add_x / 2)
    scaled_y_1 = y_1 - int(add_y / 2)
    scaled_y_2 = y_2 + int(add_y / 2)

    if scaled_y_1 < 0:
        scaled_y_1 = 0
    if scaled_y_2 > H - 1:
        scaled_y_2 = H - 1
    if scaled_x_1 < 0:
        scaled_x_1 = 0
    if scaled_x_2 > W - 1:
        scaled_x_2 = W - 1

    cropped_frames = frames[:, scaled_y_1:scaled_y_2, scaled_x_1:scaled_x_2, :]

    return cropped_frames


def Data_preprocessing(video_path, signal_path, resize_shape, scale_factor):
    subject = video_path.split('/')[-2]
    task = video_path.split('/')[-1]

    save_path = f'./Preprocessed_MTCNN_size{resize_shape[0]}_scale{scale_factor}'
    save_visualization_path = os.path.join(save_path, subject + task)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(save_visualization_path):
        os.mkdir(save_visualization_path)

    ## video pre-processing
    img_list = os.listdir(video_path)
    img_list = natsort.natsorted(img_list)
    n_frames = len(img_list)
    frames = []

    for img in img_list:
        frame_path = os.path.join(video_path, img)
        frame = cv2.imread(frame_path)
        frames.append(frame)
    frames = np.array(frames)

    cropped_frames = Video_cropping(frames, device, box_scale_factor=scale_factor)

    fps = 25
    video_dir = os.path.join(save_visualization_path, 'cropped_video_check.avi')
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    videowriter = cv2.VideoWriter(video_dir, fourcc, fps, resize_shape)

    preprocessed_video = np.zeros((cropped_frames.shape[0], resize_shape[0], resize_shape[1], cropped_frames.shape[3]),
                          dtype=np.float32)
    for n_frame in range(len(cropped_frames)):
        cropped_img = cropped_frames[n_frame, :, :, :]
        cropped_img = cv2.resize(cropped_img, (resize_shape[0], resize_shape[1]), interpolation=cv2.INTER_CUBIC)
        videowriter.write(cropped_img)
        # img unit8 → float32
        cropped_img = np.array(cropped_img).astype(np.float32)

        # normalize pixel values between 0, 1
        preprocessed_video[n_frame] = cropped_img / 255.

    videowriter.release()

    ## signal pre-processing
    raw_signal = np.loadtxt(os.path.join(signal_path))
    signal = raw_signal[:n_frames * 40]
    sampled_signal = scipy.signal.resample(signal, n_frames)
    label = sampled_signal.astype(np.float32)

    plt.figure(figsize=(20, 2), dpi=300)
    plt.plot(label, 'b-')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.tight_layout()

    plt.savefig(os.path.join(save_visualization_path, 'raw_signal_check.png'))
    plt.cla()

    plt.figure(dpi=300)
    plt.plot(label, 'b-')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    plt.tight_layout()

    plt.savefig(os.path.join(save_visualization_path, 'raw_signal_check.png'))
    plt.cla()

    #print(preprocessed_video.shape, label.shape)
    file_path = os.path.join(save_visualization_path, 'data.mat')
    mdict = {'video': preprocessed_video, 'ref': label}
    sio.savemat(file_path, mdict)


## main
# path of dataset
video_path = '/hdd1/MMSE-HR/video/'
signal_path = '/hdd1/MMSE-HR/signal/'

video_path_list = []
for (root, directories, files) in os.walk(video_path):
    for d in directories:
        if 'T' in d:
            d_path = os.path.join(root, d)
            video_path_list.append(str(d_path))

signal_path_list = []
for (root, directories, files) in os.walk(signal_path):
    for f in files:
        if 'BP_mmHg' in f:
            f_path = os.path.join(root, f)
            signal_path_list.append(str(f_path))

video_path_list = natsort.natsorted(video_path_list)
signal_path_list = natsort.natsorted(signal_path_list)

scale_factor = 1.6
resize_shape = (128, 128)
start_time = datetime.now()
for i in tqdm(range(len(video_path_list))):
     Data_preprocessing(video_path_list[i], signal_path_list[i], resize_shape, scale_factor)

duration = datetime.now() - start_time
print('Data preprocessing finished, duration : ', duration)
