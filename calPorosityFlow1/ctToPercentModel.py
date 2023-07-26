import torch
from torch import nn

# Create a convolutional neural network 
class CtToPercentModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.block = nn.Sequential(
            *block(input_shape, hidden_units, normalize=False),
            *block(hidden_units, hidden_units, normalize=False),
            *block(hidden_units, hidden_units, normalize=False),
            *block(hidden_units, hidden_units, normalize=False),
            *block(hidden_units, hidden_units, normalize=False),
            nn.Linear(hidden_units, output_shape),
        )
       
    
    def forward(self, x: torch.Tensor):
        return self.block(x)
    
    