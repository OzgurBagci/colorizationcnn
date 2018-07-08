import torch.nn as nn
import torch.utils.data as data
import torch
import numpy as np


class ColorfulCNN(nn.Module):
    def __init__(self):
        super(ColorfulCNN, self).__init__()
        self.conv0 = \
            nn.Sequential(
                nn.Conv2d(1, 4, 3, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU()
            )
        self.conv1 = \
            nn.Sequential(
                nn.Conv2d(4, 8, 3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
        self.conv2 = \
            nn.Sequential(
                nn.Conv2d(8, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        self.conv3 = \
            nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        self.conv4 = \
            nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
        self.conv5 = \
            nn.Sequential(
                nn.Conv2d(16, 8, 3, padding=1),
                nn.MaxPool2d(2),
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
        self.out = nn.Conv2d(8, 2, 3, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        return x

    @staticmethod
    def prepare_images(images):
        """
        :param images: np.array
        :return: torch.Tensor
        """

        return torch.from_numpy(np.einsum('ijkl->iljk', images))


class ColorfulDataset(data.Dataset):
    def __init__(self, ndata, target):
        super(ColorfulDataset, self).__init__()
        self.ndata = ndata
        self.target = target

    def __getitem__(self, index):
        return self.ndata[index, :, :, :], self.target[index, :, :, :]

    def __len__(self):
        return self.ndata.shape[0]
