import scipy.io as sio
from scipy.signal import argrelextrema
import scipy.signal
import interpolator as interpolator_lib
from utils import *


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


def clip_segmentation(subjects, data_path, WINDOW_SIZE, STRIDE, phase):

    WINDOW_SIZE = WINDOW_SIZE
    STRIDE = STRIDE
    DEVICE = 'cpu'

    interpolator = interpolator_lib.Interpolator(model_path='pretrained_FILM/saved_model', align=1, block_shape=[1, 1])

    for subject in subjects:
        for path in data_path:
            if subject in path:
                data = sio.loadmat(path)

                name = path.split('/')[6]

                tmp_x = data['video']
                tmp_y = np.squeeze(data['ref'])
                win_start = 0
                win_end = win_start + WINDOW_SIZE

                while win_end <= len(tmp_x):
                    tmp_value = Value_extraction(tmp_y[win_start:win_end], DEVICE)

                    aug = 'ori'
                    segment_file_path = f'/PATH OF PROJECT FILE/V4V_segments/{phase}/{name}_{win_start}_{aug}.mat'
                    mdict = {'video': tmp_x[win_start:win_end], 'ref': tmp_y[win_start:win_end], 'value': tmp_value}
                    sio.savemat(segment_file_path, mdict)

                    if phase == 'train':
                        # HRDA - temporally upsample
                        if (75 <= tmp_value[2]) & (tmp_value[2] <= 90):
                            half_y_win = tmp_y[win_start:int(win_start + WINDOW_SIZE / 2)]
                            interpolated_half_y_win = scipy.signal.resample(half_y_win, WINDOW_SIZE)
                            half_tmp_value = Value_extraction(interpolated_half_y_win, DEVICE)

                            if half_tmp_value[2] <= tmp_value[2 ] * 3 /4:

                                half_x_win = tmp_x[win_start:int(win_start +(WINDOW_SIZE / 2 + 1))]

                                upsampled = FILM(half_x_win, interpolator)

                                aug = 'up'
                                segment_file_path = f'/PATH OF PROJECT FILE/V4V_segments/{phase}/{name}_{win_start}_{aug}.mat'
                                mdict = {'video': upsampled, 'ref': interpolated_half_y_win, 'value': half_tmp_value}
                                sio.savemat(segment_file_path, mdict)

                                print("Upsampled", tmp_value[2], half_tmp_value[2])

                        # HRDA - temporally downsample
                        if (70 <= tmp_value[2]) & (tmp_value[2] <= 80):
                            double_x_win = tmp_x[win_start:int(win_start + (WINDOW_SIZE * 2))]
                            if WINDOW_SIZE * 2 <= double_x_win.shape[0]:
                                sampled_double_x_win = np.zeros((WINDOW_SIZE, double_x_win.shape[1], double_x_win.shape[2], double_x_win.shape[3]),
                                                                dtype=np.float32)
                                for frame in range(WINDOW_SIZE):
                                    sampled_double_x_win[frame, :, :, :] = double_x_win[frame*2, :, :, :]

                                double_y_win = tmp_y[win_start:int(win_start + (WINDOW_SIZE * 2 ) -1)]
                                sampled_double_y_win = scipy.signal.resample(double_y_win, WINDOW_SIZE)
                                double_tmp_value = Value_extraction(sampled_double_y_win, DEVICE)

                                aug = 'down'
                                segment_file_path = f'/PATH OF PROJECT FILE/V4V_segments/{phase}/{name}_{win_start}_{aug}.mat'
                                mdict = {'video': sampled_double_x_win, 'ref': sampled_double_y_win, 'value': double_tmp_value}
                                sio.savemat(segment_file_path, mdict)

                                print("Downsampled", tmp_value[2], double_tmp_value[2])

                    win_start += STRIDE
                    win_end += STRIDE