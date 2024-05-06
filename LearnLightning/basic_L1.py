"""
Level 1: Train a model
Learn the basics of training a model.
"""

# ref: https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html

# import 
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

# build the model 
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
    
class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define a LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
# Define the training dataset
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
model = AutoEncoder(Encoder(), Decoder())
autoencoder = LitAutoEncoder(model=model, lr = 1e-3)

# train model
trainer = L.Trainer(max_epochs=5, limit_train_batches=128)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

"""
1. how to use callback?
2. how to control L.Trainer?
3. how to use LRscheduler in L.LightningModule module?
"""
