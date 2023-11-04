import argparse
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

from rockXCTFractionalDataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--isTrain',
                    action='store_true',
                    help='does it start to training')
parser.add_argument("--dataroot",
                    type=str,
                    default='data/rockXCT_fractional',
                    help="the target of data folder")
parser.add_argument("--csv_file",
                    type=str,
                    default='data/at1_fractionalPorosity.csv',
                    help="the target of csv file")
parser.add_argument("--n_epochs",
                    type=int,
                    default=100,
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
parser.add_argument("--img_size",
                    type=int,
                    default=512,
                    help="size of each image dimension")
parser.add_argument("--channels",
                    type=int,
                    default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval",
                    type=int,
                    default=400,
                    help="interval betwen image samples")
parser.add_argument("--dataroot_pred",
                    type=str,
                    default='data/rockXCT_fractional_test',
                    help="the target of data folder to predict")
parser.add_argument("--csv_file_pred_output",
                    type=str,
                    default='normal_DL_pred_output.csv',
                    help="the target of csv file to output")
opt = parser.parse_args()


class DicomToNumberModel(L.LightningModule):
    def __init__(self, num_classes=1):
        super(DicomToNumberModel, self).__init__()
        # Use a pre-trained CNN model as the feature extractor
        self.feature_extractor = models.resnet18(pretrained=True)
        # Modify the input layer to input a single channel
        self.feature_extractor.conv1 = nn.Conv2d(1,
                                                 64,
                                                 kernel_size=(7, 7),
                                                 stride=(2, 2),
                                                 padding=(3, 3),
                                                 bias=False)
        # Modify the final classification layer to output a single number
        self.feature_extractor.fc = nn.Linear(
            self.feature_extractor.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through the network
        x = self.feature_extractor(x)
        return x

    def training_step(self, batch, batch_idx):
        # Training step
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat,
                          y.unsqueeze(1).float())  # Mean Squared Error loss
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat,
                          y.unsqueeze(1).float())  # Mean Squared Error loss
        self.log('test_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        pred = self(batch)
        return pred

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr)
        return optimizer


if opt.isTrain:

    # ----------
    #  Training
    # ----------

    dm = rockFractionalDataModule(opt.batch_size, opt.dataroot, opt.csv_file,
                                  opt.img_size, opt.channels, opt.n_cpu)

    model = DicomToNumberModel()
    trainer = L.Trainer(accelerator='auto', max_epochs=opt.n_epochs)
    trainer.fit(model, dm)

    # ----------
    #  Testing
    # ----------

    trainer.test()
else:

    # ----------
    #  Prediting
    # ----------

    dm = rockFractionalDataModule(opt.batch_size, opt.dataroot_pred,
                                  opt.csv_file_pred_output, opt.img_size,
                                  opt.channels, opt.n_cpu)

    model = DicomToNumberModel.load_from_checkpoint(
        'lightning_logs/normal_DL_fractional_100/checkpoints/epoch=99-step=24500.ckpt'
    )

    trainer = L.Trainer(accelerator='auto')

    pred_fractional_porosity = trainer.predict(model, dm)

    pred_data = np.array(pred_fractional_porosity).reshape(-1)

    output_df = pd.DataFrame(pred_data)

    output_df.to_csv(opt.csv_file_pred_output)
