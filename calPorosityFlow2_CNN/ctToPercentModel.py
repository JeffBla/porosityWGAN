import torch
from torch import nn


# VGG Block with 2 or 3 Convolutional layers
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Create a convolutional neural network
# refer VGG
class CtToPercentModel(nn.Module):
    def __init__(self,
                 input_shape=1,
                 num_conv_layers=[1, 2, 1, 1, 1, 1],
                 output_shape=3):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            VGGBlock(input_shape, 64, num_conv_layers[0]),
            VGGBlock(64, 128, num_conv_layers[1]),
            VGGBlock(128, 128, num_conv_layers[2]),
            VGGBlock(128, 128, num_conv_layers[3]),
            VGGBlock(128, 64, num_conv_layers[4]),
            VGGBlock(64, output_shape, num_conv_layers[5]))

    def forward(self, x):
        x = self.conv_blocks(x)
        return x
