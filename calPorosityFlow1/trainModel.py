import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt 

import torch.nn as nn
import torch

from targetGenerator import Generator
import dataPrepare
import trainFunc
from ctToPercentModel import CtToPercentModel


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset, test_dataset = dataPrepare.dataSet(NUM_IMAGE=100)

# Create the dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=opt.n_cpu)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=False,
                                         num_workers=opt.n_cpu)

model = CtToPercentModel(1, 16, 3)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

TrainlossList = []
TestlossList = []

# Measure time
from timeit import default_timer as timer
train_time_start_on_gpu = timer()

for epoch in tqdm(range(opt.n_epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_loss, _ = trainFunc.train_step(data_loader=train_dataloader, 
        model=model, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        device= device
    )
    test_loss, _ = trainFunc.test_step(data_loader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
        device= device
    )

    TrainlossList.append(train_loss.item())
    TestlossList.append(test_loss.item())

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu,
                                            end=train_time_end_on_gpu,
                                            device=device)

# torch.save(model.state_dict(), './model/rock/ct2Percent.pt')

plt.plot(TrainlossList, label= 'Train loss')
plt.plot(TestlossList, label= 'Test loss')
plt.show()