import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torch.autograd import Variable

import torch.nn as nn
import torch

import sys
sys.path.append('/home/jeffbla/下載/dcgan/')

from percentlayer import PercentLayer_dcgan

def batch_height_widthResize(imagePlusOneDim: torch.Tensor) -> torch.Tensor:
    output = imagePlusOneDim.view(imagePlusOneDim.shape[0], -1)
    output -= output.min(1, keepdim=True)[0]
    output /= output.max(1, keepdim=True)[0]
    output = output.view(imagePlusOneDim.shape[0], imagePlusOneDim.shape[1], imagePlusOneDim.shape[2])
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='data/rockXCT', help="the target of data folder")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
opt = parser.parse_args()

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 8

print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.feature = []
        self.iscuda = True if torch.cuda.is_available() else False
        self.batchSize = opt.batch_size

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(opt.latent_dim, ngf * 16, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. ``(ngf*16) x 4 x 4``
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 8 x 8``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 16 x 16``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 32 x 32``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), # 3 for percent layer
            # nn.Tanh(),
            PercentLayer_dcgan()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img
    
    def loadGeneraor(self):
        self.load_state_dict(torch.load('./model/rock/generator_percent.pt'))

        if self.iscuda:
            self.cuda()
    
    def set_extractPercent(self):
        self.features = []
        def hookForPercentLayerInput(module, input, output):
            self.features.append(input[0].clone().detach())

        genModules = dict(self.named_modules())
        percentLayerInGen = list(filter(lambda c: True if isinstance(c, PercentLayer_dcgan) else False,genModules.values()))[0]
        self.handel = percentLayerInGen.register_forward_hook(hookForPercentLayerInput)
    
    def generateFakeImageAndPercent(self):
        # Sample noise as generator input
        Tensor = torch.cuda.FloatTensor if self.iscuda else torch.FloatTensor
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim, 1, 1))))
        fake_imgs = self(z)

        percent = torch.softmax(self.features[0], 1)
        return fake_imgs, percent
    
    def unset_extractPercent(self):
        self.handel.remove()


