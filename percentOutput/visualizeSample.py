import sys

sys.path.append('/home/jeffbla/resourceDPProject/calPorosityFlow1')

from pydicom import dcmread
import os
import random
from pathlib import Path
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from trainFunc import rescaleCTToN1and1

NUM_PERCENT = 8

CTG = 2232.875
W = 0
AIR = -1000

DCM_PATH = Path('/home/jeffbla/下載/dcgan/data/rockXCT/bh_3_16/')

# plotly view
fig = make_subplots(3, 3, horizontal_spacing=0.01, vertical_spacing=0.01)
# read img
for i in range(NUM_PERCENT):
    hovertemplate = "x: %{x} <br> y: %{y} <br> z: %{z} <br> ct: %{customdata:.4f}"

    ds = dcmread(DCM_PATH / random.choice(os.listdir(DCM_PATH)))
    px_arr = np.array(ds.pixel_array)
    # CT value
    Hu = px_arr * ds.RescaleSlope + ds.RescaleIntercept

    img = rescaleCTToN1and1(Hu)

    Hu = Hu[:, :, np.newaxis]

    fig.add_trace(go.Heatmap(z=img, customdata=Hu,
                             hovertemplate=hovertemplate),
                  row=int(i / 3) + 1,
                  col=i % 3 + 1)

fig.show()