import typing
from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as f
from torchvision.transforms.functional import to_tensor
from torch.utils.data.dataset import Dataset
import numpy as np
import random
from tqdm import tqdm



class VAE(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> typing.Any:
        return super().__call__(*inp, **kwargs)

    def __init__(
            self,
            image_height: int = 210,
            image_width: int = 160,
            image_channels: int = 3,
            z_dim: int = 32,
            flatten_tup: tuple = (16, 25, 19)
    ):
        super().__init__()
        self.h = image_height
        self.w = image_width
        self.c = image_channels
        self.flatten_tup = flatten_tup
        self.flatten_size = np.prod(flatten_tup).item()
        self.z_dim = z_dim

        # encoder
        self.en_cnn1 = nn.Conv2d(in_channels=self.c, out_channels=32, kernel_size=(4, 4), stride=2)
        self.en_cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        self.en_cnn3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(2, 2), stride=2)

        self.en_dense_mu1 = nn.Linear(in_features=self.flatten_size, out_features=1024)
        self.en_dense_mu2 = nn.Linear(in_features=1024, out_features=self.z_dim)

        self.en_dense_s1 = nn.Linear(in_features=self.flatten_size, out_features=1024)
        self.en_dense_s2 = nn.Linear(in_features=1024, out_features=self.z_dim)

        self.de_dense1 = nn.Linear(in_features=self.z_dim, out_features=self.flatten_size)
        # resize to 25 x 19 x 16
        self.de_deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(2, 2), stride=2)
        # 50 x 38 x 32
        self.de_deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2)
        # 101 x 77 x 32
        self.de_deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=self.c, kernel_size=(10, 8), stride=2)

    def encode(self, x: torch.Tensor):
        x = x.view(-1, self.c, self.h, self.w)
        x = f.elu(self.en_cnn1(x))
        x = f.elu(self.en_cnn2(x))
        x = f.elu(self.en_cnn3(x))

        # flatten
        x = x.view(-1, self.flatten_size)

        x_mu = f.elu(self.en_dense_mu1(x))
        x_mu = f.elu(self.en_dense_mu2(x_mu))

        x_logstd = f.elu(self.en_dense_s1(x))
        x_logstd = f.elu(self.en_dense_s2(x_logstd))

        return x_mu, x_logstd

    def decode(self, x: torch.Tensor):
        x = f.elu(self.de_dense1(x))
        x = x.view(-1, *self.flatten_tup)
        x = f.elu(self.de_deconv1(x))
        x = f.elu(self.de_deconv2(x))
        x = torch.sigmoid(self.de_deconv3(x))   # image pixels are normalized to [0..1]
        return x

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        if self.training:
            std = (logstd * 0.5).exp_()
            std_prob = torch.randn(*mu.size())
            return mu + std_prob * std
        else:
            return mu   # inference time

    def forward(self, x: torch.Tensor):
        mu, logstd = self.encode(x)
        z = self.reparameterize(mu, logstd)
        z = self.decode(z)
        return z, mu, logstd

    def calculate_loss(self, pred_x: torch.Tensor, true_x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
        BCE = f.binary_cross_entropy(pred_x, true_x)
        KLD = -0.5 * torch.mean(1 + logstd - mu.pow(2) - logstd.exp())
        return BCE + KLD

    def train_model(self, images: List[np.ndarray], epochs: int = 10, batch_size: int = 32):
        opt = Adam(self.parameters())

        images = [to_tensor(x) for x in images]
        losses = []

        for i in range(epochs):
            random.shuffle(images)

            for x in tqdm(range(len(images) // batch_size), desc=f"Epoch {i + 1}/{epochs}"):
                batch = images[x * batch_size:(x + 1) * batch_size]
                batch = torch.stack(batch)

                opt.zero_grad()
                pred_x, mu, logstd = self.forward(batch)
                loss = self.calculate_loss(pred_x, batch, mu, logstd)
                loss.backward()
                opt.step()

                losses.append(loss.item())
        return losses
