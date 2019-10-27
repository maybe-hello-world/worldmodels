import random
from typing import List, Any

import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm
from torch.distributions.normal import Normal
from torch.optim import Adam


class MDNRNN(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> Any:
        return super().__call__(*inp, **kwargs)

    def __init__(self, z_dim_size: int = 32, action_size: int = 3, hidden_size: int = 256, dist_amount: int = 5):
        """
        MDN network

        :param z_dim_size: size of latent space of VAE
        :param action_size: size of action space (from Gym env)
        :param hidden_size: amount of neurons inside
        :param dist_amount: amount of modelled distributions
        """
        super().__init__()

        self.z_dim_size = z_dim_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.dist_amount = dist_amount

        self.rnn = nn.LSTM(self.z_dim_size + self.action_size, self.hidden_size)

        # mu & sigma for each z_dim_size distribution * amount of distributions + pi (distribution mix coefficients)
        self.mdn = nn.Linear(self.hidden_size, 2 * self.z_dim_size * self.dist_amount + self.dist_amount)

    def forward(self, actions: torch.Tensor, z_dim: torch.Tensor):
        seq_size, batch_size = actions.size(0), actions.size(1)

        inp = torch.cat([actions, z_dim], dim=-1)     # just glue them into one vector
        hidden_state, _ = self.rnn(inp)     # get just hidden state
        mdn_out = self.mdn(hidden_state)

        point = self.z_dim_size * self.dist_amount

        # get mus from output
        mus = mdn_out[:, :, :point]
        mus = mus.view(seq_size, batch_size, self.dist_amount, self.z_dim_size)

        # get sigmas from output
        sigmas = mdn_out[:, :, point:2*point]
        sigmas = sigmas.view(seq_size, batch_size, self.dist_amount, self.z_dim_size)
        sigmas = torch.exp(sigmas)

        # get distrib mix coefs and turn them into probs
        point = 2 * point
        pis = mdn_out[:, :, point:point+self.dist_amount]
        pis = f.log_softmax(pis, dim=-1)

        return mus, sigmas, pis

    def calculate_loss(self, true_y: torch.Tensor, mus: torch.Tensor, sigmas: torch.Tensor, pis: torch.Tensor):
        true_y = true_y.unsqueeze(-2)   # ????

        normal_dist = Normal(mus, sigmas)   # get normal dist
        g_log_probs = normal_dist.log_prob(true_y)  # get possibility of true_y
        g_log_probs = pis + torch.sum(g_log_probs, dim=-1)  # product of possibilities

        # normalize
        max_lp = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
        g_log_probs = g_log_probs - max_lp

        # get probs
        g_probs = torch.exp(g_log_probs)
        probs = torch.sum(g_probs, dim=-1)

        return -torch.mean(max_lp.squeeze() + torch.log(probs)) / self.z_dim_size

    def train_model(self, games: List[List[torch.Tensor]], epochs: int = 1) -> List[float]:
        """
        Train current model

        :param games: list of rollouts (each rollout - list of states (tensors))
        :param epochs: amount of epochs
        :return: loss history
        """
        losses = []
        opt = Adam(self.parameters())
        for i in range(epochs):
            random.shuffle(games)
            for g, rollout in enumerate(games):
                for frame in tqdm(rollout, desc=f"Epoch {i+1}/{epochs}, game {g+1}/{len(games)}"):
                    opt.zero_grad()
                    z_dim, acts, next_z_dim = frame
                    acts = torch.from_numpy(acts).float().unsqueeze(0).unsqueeze(0)
                    z_dim = z_dim.unsqueeze(0)
                    mus, sigmas, pis = self.forward(acts, z_dim)
                    loss = self.calculate_loss(next_z_dim, mus, sigmas, pis)
                    loss.backward()
                    opt.step()
                    losses.append(loss.item())
        return losses

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, *args, **kwargs):
        state_dict = torch.load(path)
        model = MDNRNN(*args, **kwargs)
        model.load_state_dict(state_dict=state_dict)
        return model
