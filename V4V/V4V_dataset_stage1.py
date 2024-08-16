from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import torch
import os
import numpy as np

class V4V_dataset_stage1(Dataset):
    def __init__(self, data_path):
        super(V4V_dataset_stage1, self).__init__()
        self.path = data_path

    def __getitem__(self, index):

        data = sio.loadmat(self.path[index])
        video_clip = torch.from_numpy(data['video']).type(torch.FloatTensor).permute(3, 0, 1, 2)
        signal_clip = torch.from_numpy(np.squeeze(data['ref'])).type(torch.FloatTensor)
        value_clip = torch.from_numpy(np.squeeze(data['value'])).type(torch.FloatTensor)

        return video_clip, signal_clip, value_clip

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    train_data_path = "/PATH OF PROJECT FILE/V4V_segments/train/"
    train_data_path_list = []
    for (root, directories, files) in os.walk(train_data_path):
        for f in files:
            if '.mat' in f:
                f_path = os.path.join(root, f)
                train_data_path_list.append(str(f_path))

    train_dataset = V4V_dataset(train_data_path_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)
    for iteration, (t_x, t_y, t_v) in enumerate(train_loader):
        print(t_x.shape, t_y.shape, t_v.shape)


