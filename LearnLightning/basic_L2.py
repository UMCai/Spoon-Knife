"""
Level 2: Add a validation and test set
Add validation and test sets to avoid over/underfitting.
"""


"""
Validate and test a model
Add a validation and test data split to avoid overfitting.

Add a test loop
* To make sure a model can generalize to an unseen dataset (ie: to publish a paper or in a production environment) 
    a dataset is normally split into two parts, the train split and the test split.
* The test set is NOT used during training, it is ONLY used once the model has been trained to see how the model will do in the real-world.
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L
import torch.utils.data as data
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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

class LitAutoEncoder(L.LightningModule):
    '''
    The training_step defines how the nn.Modules interact together.
    In the configure_optimizers define the optimizer(s) for your models.
    '''
    def __init__(self, encoder, decoder, lr):
        super().__init__()
        # to automatically save all the hyperparameters passed to init simply 
        # by calling self.save_hyperparameters().
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
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# data
transform = transforms.ToTensor()
train_set = MNIST(root=os.getcwd(), download=False, train=True, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
test_set = MNIST(root=os.getcwd(), download=False, train=False, transform=transform)


# model
autoencoder = LitAutoEncoder(Encoder(), Decoder(), lr = 1e-3)

"""
Save your model progress
Learn to save the state of a model as it trains. default_root_dir control where to save the log
"""
# train model
need_train = False
if need_train:
    """
    Enable early stopping
    Use early stopping to decide when to stop training your model.
    """
    early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
    trainer = L.Trainer(max_epochs=1, default_root_dir=os.getcwd(), enable_checkpointing=True, callbacks=[early_stop_callback])
    trainer.fit(model=autoencoder, train_dataloaders=DataLoader(train_set, batch_size=128), val_dataloaders=DataLoader(valid_set, batch_size=128))

    # Once the model has finished training, call .test
    trainer.test(model=autoencoder, dataloaders=DataLoader(test_set, batch_size=128))


load_pretrain = True
checkpoint = "./lightning_logs/version_12/checkpoints/epoch=0-step=375.ckpt"
if load_pretrain:
    model = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder = Encoder(), decoder = Decoder())
    # with the access to ckpt
    print(model.lr)
    
    # disable randomness, dropout, etc...
    model.eval()
    # predict with the model
    fake_image_batch = torch.rand(4, 28 * 28, device=model.device)
    y_hat = model.encoder(fake_image_batch)
    print(y_hat)
    
    checkpoint = torch.load(checkpoint)
    encoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
    decoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}

    print(encoder_weights)


