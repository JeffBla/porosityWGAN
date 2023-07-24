import torch
import torchvision
from torch import nn

CTG = 2232.875
W = 0
AIR= -1000

class PercentLayer_dcgan(nn.Module):
    """3 input. 1 output"""
    def forward(self, x):

        x = torch.softmax(x, 1)

        output = CTG * x[:,0] + W * x[:,1] + AIR * x[:,2]
        output = output.unsqueeze(1)
        rescale = (output-AIR) * (1/(CTG-AIR))
        rescale = 2 * rescale -1
        return rescale
