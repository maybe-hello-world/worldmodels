import random
from typing import List, Any, Tuple

import numpy as np

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

    def __init__(
            self,
            z_dim_size: int = 32,
            action_size: int = 3,
            hidden_size: int = 256,
            dist_amount: int = 5,
            device: str = "cpu"
    ):
        """
        MDN network

        :param z_dim_size: size of latent space of VAE
        :param action_size: size of action space (from Gym env)
        :param hidden_size: amount of neurons inside
        :param dist_amount: amount of modelled distributions
        """
        super().__init__()
        self.device = torch.device(device)

        self.z_dim_size = z_dim_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.dist_amount = dist_amount

        self.rnn = nn.LSTM(self.z_dim_size + self.action_size, self.hidden_size)

        # mu & sigma for each z_dim_size distribution * amount of distributions + pi (distribution mix coefficients)
        self.mdn = nn.Linear(self.hidden_size, 2 * self.z_dim_size * self.dist_amount + self.dist_amount)
        self.to(self.device)

    def forward(self, actions: torch.Tensor, z_dim: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        """
        :param actions: tensor of shape: (1, batch_size, actions_size)
        :param z_dim: tensor of shape: (1, batch_size, z_dim_size)
        :param hidden: tuple of tensors: ((1, batch_size, hidden_size), (1, batch_size, hidden_size))
        """
        seq_size, batch_size = actions.size(0), actions.size(1)

        inp = torch.cat([actions, z_dim], dim=-1)     # just glue them into one vector
        out, hidden = self.rnn(inp, hidden)     # get just hidden state
        mdn_out = self.mdn(out)

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

        return mus, sigmas, pis, hidden

    def play_predict(
            self,
            actions: np.ndarray,
            z_dim: torch.Tensor,
            hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = torch.from_numpy(actions).float().unsqueeze(0).unsqueeze(0).to(self.device)
        z_dim = z_dim.unsqueeze(0).unsqueeze(0)
        _, _, _, hidden = self.forward(actions, z_dim, hidden)
        return hidden

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

    def get_eval(self, mus: torch.Tensor, pis: torch.Tensor):
        mus = mus.squeeze().detach()
        pis = pis.squeeze().detach()
        max_pi = torch.argmax(pis)
        max_mu = mus[max_pi]
        return max_mu

    def init_state(self, batch_size: int = 1):
        return (
            torch.zeros((1, batch_size, self.hidden_size)).to(self.device),
            torch.zeros((1, batch_size, self.hidden_size)).to(self.device)
        )

    def train_model(self, games: List[List[torch.Tensor]], epochs: int = 1, batch_size: int = 1) -> List[float]:
        """
        Train current model

        :param games: list of rollouts (each rollout - list of states (tensors))
        :param epochs: amount of epochs
        :param batch_size: batch size for rnn
        :return: loss history
        """
        losses = []
        opt = Adam(self.parameters())
        for i in range(epochs):
            random.shuffle(games)
            for g in range(len(games) // batch_size):

                # oh shit: rollouts to batch rollouts + unsqueeze + to cuda
                batch_rollout = games[g * batch_size:(g+1)*batch_size]
                batch_rollout = [
                    [d[b] for d in batch_rollout]
                    for b
                    in range(len(batch_rollout[0]))
                ]
                batch_rollout = [
                    (
                        (torch.stack([c[0] for c in b], dim=-2).to(self.device)),
                        (torch.stack([torch.from_numpy(c[1]).float().unsqueeze(0) for c in b], dim=-2).to(self.device)),
                        (torch.stack([c[2] for c in b], dim=-2)).to(self.device)
                     )
                    for b
                    in batch_rollout
                ]
                # end of oh shit

                hidden = (
                    torch.zeros((1, batch_size, self.hidden_size)).to(self.device),
                    torch.zeros((1, batch_size, self.hidden_size)).to(self.device)
                )
                for frame in tqdm(batch_rollout, desc=f"Epoch {i+1}/{epochs}, game {g+1}/{len(games) // batch_size}"):
                    opt.zero_grad()
                    z_dim, acts, next_z_dim = frame
                    mus, sigmas, pis, hidden = self.forward(acts, z_dim, hidden)
                    loss = self.calculate_loss(next_z_dim, mus, sigmas, pis)
                    loss.backward()
                    opt.step()
                    hidden = (hidden[0].detach(), hidden[1].detach())
                    losses.append(loss.cpu().item())
        return losses

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, *args, **kwargs):
        state_dict = torch.load(path)
        model = MDNRNN(*args, **kwargs)
        model.load_state_dict(state_dict=state_dict)
        return model
