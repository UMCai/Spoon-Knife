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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

# Define the PyTorch nn.Modules, one encoder and one decoder
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
    def __init__(self, encoder, decoder, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
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

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

  
# Define the training dataset
"""
Define a PyTorch DataLoader which contains your training dataset.
"""
# change your data path here
data_path = r"C:\Users\Shizh\OneDrive - Maastricht University\Data"
# Load data sets
transform = transforms.ToTensor()
train_set = MNIST(root=data_path, download=False, train=True, transform=transform)
test_set = MNIST(root=data_path, download=False, train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size
# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
train_loader = DataLoader(train_set, pin_memory=True, batch_size=256)
valid_loader = DataLoader(valid_set, pin_memory=True, batch_size=256)

# Train the model
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder(), lr = 1e-3)
#checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
#print(checkpoint["hyper_parameters"])

# train model
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max")
checkpoint_callback = ModelCheckpoint(dirpath='.') # i did not use it here
trainer = L.Trainer(max_epochs=3, default_root_dir='.', callbacks=[early_stop_callback], profiler="simple" )

# this is for training and validation
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)
# this is for testing (real life inference)
trainer.test(model=autoencoder, dataloaders=DataLoader(test_set, batch_size=256))
# Eliminate the training loop
"""
Under the hood, the Lightning Trainer runs the following training loop on your behalf

autoencoder = LitAutoEncoder(Encoder(), Decoder())
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

The power of Lightning comes when the training loop gets complicated as you add validation/test splits, schedulers, 
distributed training and all the latest SOTA techniques.
With Lightning, you can add mix all these techniques together without needing to rewrite a new loop every time.    
"""

# LightningModule from checkpoint
"""
model = LitAutoEncoder.load_from_checkpoint("/path/to/checkpoint.ckpt")
print(model.eval())
"""