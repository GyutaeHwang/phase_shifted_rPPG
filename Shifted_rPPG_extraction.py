import torch.nn as nn
import torch
from torchinfo import summary


class Shifted_rPPG_extraction(nn.Module):
    def __init__(self):
        super(Shifted_rPPG_extraction, self).__init__()

        self.Feature_extracter = nn.Sequential(
            nn.Conv3d(3, 16, [3,5,5],stride=1, padding=(4,2,2), dilation=(4,1,1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),

            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=(4,1,1), dilation=(4,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=(4,1,1), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
        )

        self.DilatedConv = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 2, 2], stride=1, padding=(4,0,0), dilation=(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.Attention_temporal = nn.Sequential(
            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 1, [1, 1, 1], stride=1, padding=0),

            nn.Sigmoid()
        )

        self.Attention_spatial = nn.Sequential(
            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool3d((4, 2, 2), stride=(4, 2, 2)),

            nn.Conv3d(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.AvgPool3d((2, 1, 1), stride=(2, 1, 1)),

            nn.Conv3d(32, 1, [1, 1, 1], stride=1, padding=0),

            nn.Sigmoid()
        )

        self.finger_rPPG_extraction = nn.Sequential(
            nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 1, [1, 1, 1],stride=1, padding=0),
        )

        self.face_rPPG_extraction = nn.Sequential(
            nn.AvgPool3d((1, 4, 4), stride=(1, 4, 4)),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [3, 1, 1], stride=1, padding=(4, 0, 0), dilation=(4,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 64, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 32, [1, 1, 1], stride=1, padding=0),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 1, [1, 1, 1],stride=1, padding=0),
        )

    def forward(self, x1):

        x2 = self.Feature_extracter(x1)
        x3 = self.DilatedConv(x2)

        At = self.Attention_temporal(x2)
        x3_At = torch.mul(x3, At)

        As = self.Attention_spatial(x2)
        x3_Ats = torch.mul(x3_At, As)

        face_rPPG = self.face_rPPG_extraction(x3_Ats)
        face_rPPG = face_rPPG.view(face_rPPG.shape[0], face_rPPG.shape[2]) # [B, 128]

        finger_rPPG = self.finger_rPPG_extraction(x3_Ats)
        finger_rPPG = finger_rPPG.view(finger_rPPG.shape[0], finger_rPPG.shape[2])

        return face_rPPG, finger_rPPG


