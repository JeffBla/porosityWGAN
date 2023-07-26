# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkIOImage import vtkDICOMImageReader

import argparse
import os
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from pathlib import Path

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.utils as vutils

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from percentlayer import PercentLayer_dcgan

def batch_height_widthResize(imagePlusOneDim: torch.Tensor) -> torch.Tensor:
    output = imagePlusOneDim.view(imagePlusOneDim.shape[0], -1)
    output -= output.min(1, keepdim=True)[0]
    output /= output.max(1, keepdim=True)[0]
    output = output.view(imagePlusOneDim.shape[0], imagePlusOneDim.shape[1], imagePlusOneDim.shape[2])
    return output


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='data/rockXCT', help="the target of data folder")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 8

print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print(f'cuda: {cuda}')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # input is ``(nc) x 128 x 128``
            nn.Conv2d(opt.channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 64 x 64``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 32 x 32``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 16 x 16``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 8 x 8``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*16) x 4 x 4``
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            )

    def forward(self, img):
        validity = self.model(img)
        validity = validity.view(validity.shape[0],-1)
        return validity


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator()
generator.load_state_dict(torch.load('./model/rock/generator_percent.pt'))
discriminator = Discriminator()
discriminator.load_state_dict(torch.load('./model/rock/discriminator_percent.pt'))

# extract percent
features = []
def hookForPercentLayerInput(module, input, output):
    features.append(input[0].clone().detach())

genModules = dict(generator.named_modules())
percentLayerInGen = list(filter(lambda c: True if isinstance(c, PercentLayer_dcgan) else False,genModules.values()))[0]
handel = percentLayerInGen.register_forward_hook(hookForPercentLayerInput)

if cuda:
    generator.cuda()
    discriminator.cuda()

class rockXCTDicomDataset(Dataset):
    def __init__(self, ct_imgSet, transform=None):
        self.ct_imgSet = ct_imgSet
        self.transform = transform

    def __len__(self):
        return self.ct_imgSet.shape[0]

    def __getitem__(self, idx):
        image = self.ct_imgSet[idx]
        if self.transform:
            image = self.transform(image)
        return image

datasetList = [] 
dataroot = Path(opt.dataroot)
for folder in os.listdir(dataroot):
    targetFolder = dataroot/folder
    reader = vtkDICOMImageReader()
    reader.SetDirectoryName(str(targetFolder))
    reader.Update()

    files = os.listdir(targetFolder)

    dcmImage_CT = np.array(reader.GetOutput().GetPointData().GetScalars()).reshape(
        len(files), reader.GetHeight(), reader.GetWidth())

    dcmImage_CT_tensor = torch.tensor(dcmImage_CT, dtype=torch.float)
    batch_height_widthResize(dcmImage_CT_tensor)
    dcmImage_CT_tensor = dcmImage_CT_tensor.unsqueeze(1)

    dataset = rockXCTDicomDataset(ct_imgSet=dcmImage_CT_tensor,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.img_size, antialias=False),
                                    transforms.Normalize((0.5),
                                                        (0.5)),
                                                        ]))
    datasetList.append(dataset)
wholeDataset = torch.utils.data.ConcatDataset(datasetList)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(wholeDataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=2)

# Sample noise as generator input
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim, 1, 1))))
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))
fake_imgs = generator(z)

print(torch.softmax(features[0], 1))

handel.remove()

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(real_batch[0][:64],
                         padding=5,
                         normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(
    np.transpose(
        vutils.make_grid(fake_imgs.data[:25],
                         padding=5,
                         nrow=5, 
                         normalize=True).cpu(), (1, 2, 0)))
plt.show()

from PIL import Image

fake_imgs_np = fake_imgs[1].cpu().detach().squeeze().numpy()
img = Image.fromarray(fake_imgs_np * 255)
img.convert('RGB').save('./output.tiff', 'TIFF')