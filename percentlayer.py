import lightning as L
import pytorch_lightning as pl

import torch
from torch import nn

CTG = 2232.875
W = 0
AIR = -1000


class PercentLayer_dcgan(nn.Module):
    """3 input. 1 output"""
    def forward(self, x):

        x = torch.softmax(x, 1)

        output = CTG * x[:, 0] + W * x[:, 1] + AIR * x[:, 2]
        output = output.unsqueeze(1)
        rescale = (output - AIR) * (1 / (CTG - AIR))
        rescale = 2 * rescale - 1
        return rescale


class PercentLayer_lightning(pl.LightningModule):
    """
        x: 3 * int(np.prod(self.img_shape)) 
        img_shape -> (channels, width, height)
    """
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape

    def forward(self, x):

        x = x.view([x.shape[0], 3, self.img_shape[1], self.img_shape[2]])
        x = torch.softmax(x, 1)

        output = CTG * x[:, 0] + W * x[:, 1] + AIR * x[:, 2]
        output = output.unsqueeze(1)
        rescale = (output - AIR) * (1 / (CTG - AIR))
        rescale = 2 * rescale - 1
        return rescale.view(rescale.shape[0], -1)