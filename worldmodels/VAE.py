import typing
from typing import List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import functional as f
from torchvision.transforms.functional import to_tensor
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
            image_height: int = 56,
            image_width: int = 64,
            image_channels: int = 3,
            z_dim: int = 32,
            device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.h = image_height
        self.w = image_width
        self.c = image_channels
        self.z_dim = z_dim

        self.en_cnn1 = nn.Conv2d(in_channels=self.c, out_channels=32, kernel_size=3, stride=2)
        self.en_cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
        self.en_cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1)

        self.en_dense_mu1 = nn.Linear(in_features=21504, out_features=self.z_dim)
        self.en_dense_s1 = nn.Linear(in_features=21504, out_features=self.z_dim)

        self.de_dense1 = nn.Linear(in_features=self.z_dim, out_features=12544)
        self.de_deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2, 4), stride=2)
        self.de_deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.de_deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=self.c, kernel_size=2, stride=2)

        self.to(self.device)

    def encode(self, x: torch.Tensor):
        x = x.to(self.device)
        x = f.relu(self.en_cnn1(x))
        x = f.relu(self.en_cnn2(x))
        x = f.relu(self.en_cnn3(x))
        x = x.view(x.size(0), -1)

        mu = self.en_dense_mu1(x)
        logstd = self.en_dense_s1(x)

        return mu, logstd

    def decode(self, x: torch.Tensor):
        x = x.to(self.device)
        x = f.relu(self.de_dense1(x))
        x = x.view(-1, 256, 7, 7)
        x = f.relu(self.de_deconv1(x))
        x = f.relu(self.de_deconv2(x))
        x = torch.sigmoid(self.de_deconv3(x))
        return x

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor):
        if self.training:
            std = (logstd * 0.5).exp_()
            std_prob = torch.randn(*mu.size())
            return mu + std_prob * std
        else:
            return mu   # inference time

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        mu, logstd = self.encode(x)
        z = self.reparameterize(mu, logstd)
        z = self.decode(z)
        return z, mu, logstd

    @staticmethod
    def calculate_loss(pred_x: torch.Tensor, true_x: torch.Tensor, mu: torch.Tensor, logstd: torch.Tensor):
        bce = f.mse_loss(pred_x, true_x, size_average=False)
        kld = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
        return bce + kld

    def train_model(self, images: List[np.ndarray], epochs: int = 10, batch_size: int = 32):
        opt = Adam(self.parameters())

        images = [to_tensor(x) for x in images]
        losses = []

        for i in range(epochs):
            random.shuffle(images)

            for x in tqdm(range(len(images) // batch_size), desc=f"Epoch {i + 1}/{epochs}"):
                batch = images[x * batch_size:(x + 1) * batch_size]
                batch = torch.stack(batch).float()

                opt.zero_grad()
                pred_x, mu, logstd = self.forward(batch)
                loss = self.calculate_loss(pred_x, batch, mu, logstd)
                loss.backward()
                opt.step()

                losses.append(loss.cpu().item())
        return losses

    def play_encode(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            obs = to_tensor(obs).unsqueeze(0).float()
            return self.reparameterize(*self.encode(obs)).squeeze()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, *args, **kwargs):
        state_dict = torch.load(path)
        vae = VAE(*args, **kwargs)
        vae.load_state_dict(state_dict=state_dict)
        return vae

