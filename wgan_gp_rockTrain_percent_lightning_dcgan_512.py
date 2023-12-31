# noinspection PyUnresolvedReferences
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS

import argparse
import os
import numpy as np

import torchvision.utils as vutils

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

import lightning as L

from percentlayer import PercentLayer_dcgan
from rockXCTDicDataset import *


def batch_height_widthRescale(imagePlusOneDim: torch.Tensor) -> torch.Tensor:
    output = imagePlusOneDim.view(imagePlusOneDim.shape[0], -1)
    output -= output.min(1, keepdim=True)[0]
    output /= output.max(1, keepdim=True)[0]
    output = output.view(imagePlusOneDim.shape[0], imagePlusOneDim.shape[1],
                         imagePlusOneDim.shape[2])
    return output


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot",
                    type=str,
                    default='data/rockXCT',
                    help="the target of data folder")
parser.add_argument("--n_epochs",
                    type=int,
                    default=20,
                    help="number of epochs of training")
parser.add_argument("--batch_size",
                    type=int,
                    default=8,
                    help="size of the batches")
parser.add_argument("--lr",
                    type=float,
                    default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1",
                    type=float,
                    default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2",
                    type=float,
                    default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=4,
    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim",
                    type=int,
                    default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size",
                    type=int,
                    default=512,
                    help="size of each image dimension")
parser.add_argument("--channels",
                    type=int,
                    default=1,
                    help="number of image channels")
parser.add_argument("--n_critic",
                    type=int,
                    default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value",
                    type=float,
                    default=0.01,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval",
                    type=int,
                    default=400,
                    help="interval betwen image samples")
opt = parser.parse_args()

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_dim, ngf * 64, 4, 1, 0, bias=False),
            nn.ReLU(True),
            # state size. ``(ngf*16) x 4 x 4``
            nn.ConvTranspose2d(ngf * 64, ngf * 32, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. ``(ngf*16) x 8 x 8``
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. ``(ngf*16) x 16 x 16``
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 32 x 32``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 64 x 64``
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 128 x 128``
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. ``(3) x 256 x 256``
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1,
                               bias=False),  # 3 for percent layer
            PercentLayer_dcgan()
            # state size. ``(nc) x 512 x 512``
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            # input is ``(nc) x 512 x 512``
            nn.Conv2d(img_shape[0], ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 256 x 256``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 128 x 128``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 64 x 64``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 32 x 32``
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*16) x 16 x 16``
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*32) x 8 x 8``
            nn.Conv2d(ndf * 32, ndf * 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*64) x 4 x 4``
            nn.Conv2d(ndf * 64, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img):
        validity = self.model(img)
        validity = validity.view(validity.shape[0], -1)
        return validity


class WGAN_gp(L.LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = opt.latent_dim,
        lr: float = opt.lr,
        b1: float = opt.b1,
        b2: float = opt.b2,
        batch_size: int = opt.batch_size,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim,
                                   img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(self.hparams.batch_size,
                                        self.hparams.latent_dim, 1, 1)

        self.example_input_array = torch.zeros(self.hparams.batch_size,
                                               self.hparams.latent_dim, 1, 1)

        self.lambda_gp = 10

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self,
                         real_validity=None,
                         fake_validity=None,
                         gradient_penalty=0):
        if real_validity is None:
            # g_loss
            return -torch.mean(fake_validity)
        else:
            # d_loss
            loss = -torch.mean(real_validity) + torch.mean(
                fake_validity) + self.lambda_gp * gradient_penalty
        return loss

    def training_step(self, batch, batch_idx):
        imgs = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim, 1, 1)
        z = z.type_as(imgs)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # Real img
        real_imgs = imgs
        # Fake images
        fake_imgs = self(z)
        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(
            self.discriminator, real_imgs.data, fake_imgs.data)

        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs.detach())

        # discriminator loss is the average of these
        d_loss = self.adversarial_loss(real_validity, fake_validity,
                                       gradient_penalty)
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Train the generator every n_critic steps
        if batch_idx % opt.n_critic == 0:
            # train generator
            # generate images
            self.toggle_optimizer(optimizer_g)
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = vutils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            fake_validity = self.discriminator(self.generated_imgs)
            g_loss = self.adversarial_loss(fake_validity=fake_validity)
            self.log("g_loss", g_loss, prog_bar=True)
            self.manual_backward(g_loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(),
                                 lr=lr,
                                 betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=lr,
                                 betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = vutils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid,
                                         self.current_epoch)

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples +
                        ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(torch.ones((real_samples.shape[0], 1),
                                   dtype=torch.float,
                                   device=self.device),
                        requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
        return gradient_penalty


# ----------
#  Training
# ----------

dm = rockDataModule(opt.batch_size, opt.dataroot, opt.img_size, opt.channels, opt.n_cpu)
model = WGAN_gp(*dm.dim)
trainer = L.Trainer(accelerator='auto', max_epochs=60)
trainer.fit(model, dm)
