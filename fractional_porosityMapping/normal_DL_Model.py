import argparse
from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
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
                    default='data/rockXCT_fractional_clip',
                    help="the target of data folder")
parser.add_argument("--csv_file",
                    type=str,
                    default='data/fractionalPorosity.csv',
                    help="the target of csv file")
parser.add_argument("--n_epochs",
                    type=int,
                    default=10,
                    help="number of epochs of training")
parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="size of the batches")
parser.add_argument("--lr",
                    type=float,
                    default=0.02,
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
                    default='data/rockXCT_fractional_pred',
                    help="the target of data folder to predict")
parser.add_argument("--csv_file_pred_output",
                    type=str,
                    default='normal_DL_pred_output.csv',
                    help="the target of csv file to output")
parser.add_argument("--prev_ckpt_path",
                    type=str,
                    default=None,
                    help="load previous checkpoint for further training")
opt = parser.parse_args()


class DicomToNumberModel(L.LightningModule):
    def __init__(self, num_classes=1, dm: rockFractionalDataModule = None):
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

        self.dm = dm

    def forward(self, x):
        # Forward pass through the network
        x = self.feature_extractor(x)
        return x

    def training_step(self, batch, batch_idx):
        # Training step
        self.train()
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(
            y_hat,
            y.unsqueeze(1).float()) * 100000  # Mean Squared Error loss
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any,
                           batch_idx: int) -> None:
        if batch_idx % 100 == 0:
            self.eval()
            with torch.no_grad():
                test_dataloader = self.dm.test_dataloader()
                for test_batch in test_dataloader:
                    x, y = test_batch
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self(x)
                    loss = F.mse_loss(y_hat, y.unsqueeze(1).float()) * 100000
                    self.log('test_loss',
                             loss,
                             on_step=True,
                             on_epoch=True,
                             prog_bar=True,
                             logger=True)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(
            y_hat,
            y.unsqueeze(1).float()) * 100000  # Mean Squared Error loss
        self.log('final_test_loss',
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

    model = DicomToNumberModel(dm=dm)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        monitor='train_loss',
        filename='resnet-{epoch:02d}-{train_loss:.4f}')
    trainer = L.Trainer(accelerator='auto',
                        max_epochs=opt.n_epochs,
                        callbacks=[checkpoint_callback])
    trainer.fit(model, dm, ckpt_path=opt.prev_ckpt_path)

    # ----------
    #  Testing
    # ----------

    trainer.test(model, dm)
else:

    # ----------
    #  Prediting
    # ----------

    dm = rockFractionalDataModule(1, opt.dataroot_pred,
                                  opt.csv_file_pred_output, opt.img_size,
                                  opt.channels, opt.n_cpu)

    model = DicomToNumberModel.load_from_checkpoint(
        'lightning_logs/resnet101_1000_clip/checkpoints/resnet-epoch=983-train_loss=3.2695.ckpt'
    )

    trainer = L.Trainer(accelerator='auto')

    pred_fractional_porosity = trainer.predict(model, dm)

    pred_data = np.array(pred_fractional_porosity).reshape(-1)

    output_df = pd.DataFrame(pred_data)

    output_df.to_csv(opt.csv_file_pred_output)
