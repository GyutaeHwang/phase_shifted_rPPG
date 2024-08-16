from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
import torch
import os
import numpy as np

class V4V_dataset_stage2(Dataset):
    def __init__(self, data_path):
        super(V4V_dataset_stage2, self).__init__()
        self.path = []

        for path in data_path:
            data = sio.loadmat(path)
            value_clip = torch.from_numpy(np.squeeze(data['value'])).type(torch.FloatTensor)
            if value_clip[0] <= 80 or 160 <= value_clip[0]:
                pass

            elif value_clip[1] <= 40 or 100 <= value_clip[1]:
                pass

            else:
                self.path.append(path)

    def __getitem__(self, index):

        data = sio.loadmat(self.path[index])
        facial_clip = torch.from_numpy(np.squeeze(data['facial'])).type(torch.FloatTensor)
        acral_clip = torch.from_numpy(np.squeeze(data['acral'])).type(torch.FloatTensor)
        ref_clip = torch.from_numpy(np.squeeze(data['ref'])).type(torch.FloatTensor)
        value_clip = torch.from_numpy(np.squeeze(data['value'])).type(torch.FloatTensor)

        return facial_clip, acral_clip, ref_clip, value_clip

    def __len__(self):
        return len(self.path)


if __name__ == '__main__':
    train_data_path = "/PATH OF PROJECT FILE/V4V_stage1_segments/train/"
    train_data_path_list = []
    for (root, directories, files) in os.walk(train_data_path):
        for f in files:
            if '.mat' in f:
                f_path = os.path.join(root, f)
                train_data_path_list.append(str(f_path))

    train_dataset = V4V_dataset_stage2(train_data_path_list)
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4)
    for iteration, (facial, acral, t_y, t_v) in enumerate(train_loader):
        print(facial.shape, acral.shape, t_y.shape, t_v.shape)


