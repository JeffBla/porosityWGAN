# noinspection PyUnresolvedReferences
from typing import Any, Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS

import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkIOImage import vtkDICOMImageReader

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset

import lightning as L

CTG = 3000
AIR = -1024


def removeDcmImage(dcmImage_CT_tensor, indexes):
    preserve_indexes = range(dcmImage_CT_tensor.shape[0])
    preserve_indexes = list(
        filter((lambda x: preserve_indexes[x] not in indexes),
               preserve_indexes))
    return dcmImage_CT_tensor[preserve_indexes]


def batch_height_widthRescale(imagePlusOneDim: torch.Tensor) -> torch.Tensor:
    return (imagePlusOneDim - AIR) / (CTG - AIR)


# this method should be wrong
def batch_height_widthRescale_withImgMaxMin(
        imagePlusOneDim: torch.Tensor) -> torch.Tensor:
    output = imagePlusOneDim.view(imagePlusOneDim.shape[0], -1)
    output -= output.min(1, keepdim=True)[0]
    output /= output.max(1, keepdim=True)[0]
    output = output.view(*imagePlusOneDim.shape)
    return output


class rockXCTFractionMappingDataset(Dataset):
    def __init__(self, fractional_porositySet, ct_imgSet, transform=None):
        self.ct_imgSet = ct_imgSet
        self.fractional_porositySet = fractional_porositySet
        self.transform = transform

    def __len__(self):
        return self.ct_imgSet.shape[0]

    def __getitem__(self, idx):
        image = self.ct_imgSet[idx]

        if self.transform:
            image = self.transform(image)

        if self.fractional_porositySet is None:
            return image
        else:
            return image, self.fractional_porositySet[idx]


class rockFractionalDataModule(L.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 dataroot: str = None,
                 csv_file: str = None,
                 img_size: int = 512,
                 channel: int = 1,
                 n_cpu: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.dataroot = dataroot
        self.csv_file = csv_file
        self.datasetList = []
        self.img_size = img_size
        self.n_cpu = n_cpu
        self.dim = (channel, self.img_size, self.img_size)
        self.train_test_spilt = 0.7

    def setup(self, stage):
        if stage == 'predict':
            reader = vtkDICOMImageReader()
            reader.SetDirectoryName(self.dataroot)
            reader.Update()

            dcmImage_CT_tensor = self.ReadDcmFolder(self.dataroot, reader)

            # Create Dataset
            self.pred_dataset = rockXCTFractionMappingDataset(
                fractional_porositySet=None,
                ct_imgSet=dcmImage_CT_tensor,
                transform=transforms.Compose([
                    transforms.Resize(self.img_size, antialias=False),
                    transforms.Normalize((0.5), (0.5)),
                ]))

        else:
            csvPorosityRef = pd.read_csv(self.csv_file)

            datarootPath = Path(self.dataroot)
            for folder in os.listdir(datarootPath):
                targetFolder = datarootPath / folder
                reader = vtkDICOMImageReader()
                reader.SetDirectoryName(str(targetFolder))
                reader.Update()

                dcmImage_CT_tensor = self.ReadDcmFolder(targetFolder, reader)

                # attach fractional porosity
                # resolve the target by file path
                core, section, group = folder.split('_')[:3]
                section = int(section)
                group = int(group)
                fractionalPorosity = csvPorosityRef[
                    (csvPorosityRef['Core ID'] == core)
                    & (csvPorosityRef['Section (m)'] == section) &
                    (csvPorosityRef['Group'] == group)]['Fractional porosity']

                target_df_values = fractionalPorosity.values

                # Create Dataset
                dataset = rockXCTFractionMappingDataset(
                    fractional_porositySet=target_df_values,
                    ct_imgSet=dcmImage_CT_tensor,
                    transform=transforms.Compose([
                        transforms.Resize(self.img_size, antialias=False),
                        transforms.Normalize((0.5), (0.5)),
                    ]))
                self.datasetList.append(dataset)

            self.wholeDataset = ConcatDataset(self.datasetList)

            generator = torch.Generator().manual_seed(42)
            self.train_data, self.test_data = random_split(
                self.wholeDataset,
                [self.train_test_spilt, 1 - self.train_test_spilt], generator)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_cpu)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.pred_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_cpu)

    def ReadDcmFolder(self, folder,
                      reader: vtkDICOMImageReader) -> torch.Tensor:
        files = os.listdir(folder)

        dcmImage_CT = np.array(
            reader.GetOutput().GetPointData().GetScalars()).reshape(
                len(files), reader.GetHeight(), reader.GetWidth())

        dcmImage_CT_tensor = torch.tensor(dcmImage_CT, dtype=torch.float)
        dcmImage_CT_tensor = dcmImage_CT_tensor.unsqueeze(1)
        dcmImage_CT_tensor = batch_height_widthRescale_withImgMaxMin(
            dcmImage_CT_tensor)

        return dcmImage_CT_tensor
