import sys

sys.path.append('/home/jeffbla/resourceDPProject/model')
sys.path.append('/home/jeffbla/resourceDPProject/dcmCutCycleOut_copy')

import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch.nn as nn
import torch

from trainFunc import rescaleCTToN1and1
from ctToPercentModel import CtToPercentModel
from readDicomCntCT_findInnerCircle_findPorosityByVTK_Output import getTargetCtDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="size of the batches")
parser.add_argument(
    "--n_cpu",
    type=int,
    default=4,
    help="number of cpu threads to use during batch generation")
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = CtToPercentModel()

model.load_state_dict(torch.load('./model/ct2Percent/ct2PercentCNN.pt'))

targetHuList = getTargetCtDataset('./dcmCutCycleOut_copy')

model.eval()
with torch.inference_mode():
    porosityList = np.array([])
    for seqCt in targetHuList:
        seqPorosity = np.array([])
        for val in seqCt:
            val = rescaleCTToN1and1(val)
            val = torch.tensor(val).unsqueeze(0).type(torch.float32)
            sPercent = model(val.unsqueeze(0))[0, 0]
            seqPorosity = np.append(seqPorosity, 1 - sPercent.detach().numpy())
        porosityList = np.append(porosityList,
                                 seqPorosity.sum() / len(seqPorosity))
print(porosityList)