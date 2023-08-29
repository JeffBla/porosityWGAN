# noinspection PyUnresolvedReferences
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
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

from torch.utils.data import DataLoader, Dataset

import torch

import lightning as L


def batch_height_widthRescale(imagePlusOneDim: torch.Tensor) -> torch.Tensor:
    output = imagePlusOneDim.view(imagePlusOneDim.shape[0], -1)
    output -= output.min(1, keepdim=True)[0]
    output /= output.max(1, keepdim=True)[0]
    output = output.view(imagePlusOneDim.shape[0], imagePlusOneDim.shape[1],
                         imagePlusOneDim.shape[2])
    return output


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


class rockDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 dataroot: str = None,
                 img_size: int = 512,
                 channel: int = 1,
                 n_cpu: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.dataroot = dataroot
        self.datasetList = []
        self.img_size = img_size
        self.n_cpu = n_cpu
        self.dim = (channel, self.img_size, self.img_size)

    def setup(self, stage):
        datarootPath = Path(self.dataroot)
        for folder in os.listdir(datarootPath):
            targetFolder = datarootPath / folder
            reader = vtkDICOMImageReader()
            reader.SetDirectoryName(str(targetFolder))
            reader.Update()

            files = os.listdir(targetFolder)

            dcmImage_CT = np.array(
                reader.GetOutput().GetPointData().GetScalars()).reshape(
                    len(files), reader.GetHeight(), reader.GetWidth())

            dcmImage_CT_tensor = torch.tensor(dcmImage_CT, dtype=torch.float)
            batch_height_widthRescale(dcmImage_CT_tensor)
            dcmImage_CT_tensor = dcmImage_CT_tensor.unsqueeze(1)

            dataset = rockXCTDicomDataset(
                ct_imgSet=dcmImage_CT_tensor,
                transform=transforms.Compose([
                    transforms.Resize(self.img_size, antialias=False),
                    transforms.Normalize((0.5), (0.5)),
                ]))
            self.datasetList.append(dataset)

        self.wholeDataset = torch.utils.data.ConcatDataset(self.datasetList)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.wholeDataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_cpu)