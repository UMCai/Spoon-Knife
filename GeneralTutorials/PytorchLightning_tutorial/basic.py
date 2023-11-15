"""
Pytorch lightning basic skills tutorial :)
this tutorial is totally based on https://lightning.ai/docs/pytorch/stable/levels/core_skills.html
let's start!
"""

####### TRAIN A MODEL
# Add imports
"""
Add the relevant imports at the top of the file
"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

# Define the PyTorch nn.Modules
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
    
# Define a LightningModule
"""
The LightningModule is the full recipe that defines how your nn.Modules interact.
    * The training_step defines how the nn.Modules interact together.
    * In the configure_optimizers define the optimizer(s) for your models.
"""
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# Define the training dataset
"""
Define a PyTorch DataLoader which contains your training dataset.
"""
