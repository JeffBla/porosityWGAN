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
from torch.utils.data import DataLoader, Dataset, random_split

import lightning as L


def removeDcmImage(dcmImage_CT_tensor, index):
    return torch.cat(
        (dcmImage_CT_tensor[:index], dcmImage_CT_tensor[index + 1:]))


def batch_height_widthRescale(imagePlusOneDim: torch.Tensor) -> torch.Tensor:
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

        sample = {
            'image': image,
            'fractional_porosity': self.fractional_porositySet[idx]
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


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
        fractional_df = pd.read_csv(self.csv_file)

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
            dcmImage_CT_tensor = dcmImage_CT_tensor.unsqueeze(1)
            dcmImage_CT_tensor = batch_height_widthRescale(dcmImage_CT_tensor)

            # fractional porosity appending
            section_num = int(folder.split('_')[-1])
            section_df = fractional_df[fractional_df['Section (m)'] ==
                                       section_num]
            section_df = section_df[:dcmImage_CT_tensor.shape[0]].reset_index()

            # remove nan along with dcm image
            remove_indexes = section_df[
                section_df['Fractional porosity'].isna()].index

            # remove value inside dcmImage tensor
            for i in remove_indexes:
                dcmImage_CT_tensor = removeDcmImage(dcmImage_CT_tensor, i)

            # remove valuse inside dataFrame
            section_df.dropna(inplace=True)

            # Create Dataset
            dataset = rockXCTFractionMappingDataset(
                fractional_porositySet=section_df['Fractional porosity'].
                values,
                ct_imgSet=dcmImage_CT_tensor,
                transform=transforms.Compose([
                    transforms.Resize(self.img_size, antialias=False),
                    transforms.Normalize((0.5), (0.5)),
                ]))
            self.datasetList.append(dataset)

        self.wholeDataset = torch.utils.data.ConcatDataset(self.datasetList)

        self.train_data, self.test_data = random_split(
            self.wholeDataset,
            [self.train_test_spilt, 1 - self.train_test_spilt])

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
