from typing import Any, Tuple
from collections import deque
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

from utils.noise import OrnsteinUhlenbeckActionNoise


class Critic(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> Any:
        return super().__call__(*inp, **kwargs)

    def __init__(self, state_dim: int = 32, hidden_dim: int = 256, action_dim: int = 3, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        self.fc1 = nn.Linear(state_dim + hidden_dim + action_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        x = f.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Actor(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> Any:
        return super().__call__(*inp, **kwargs)

    def __init__(self, state_dim: int = 32, hidden_dim: int = 256, action_dim: int = 3, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.fc1 = nn.Linear(state_dim + hidden_dim, 32)
        self.fc2 = nn.Linear(32, action_dim)

    def forward(self, state: torch.Tensor):
        x = f.leaky_relu(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return x


class Controller(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> Any:
        return super().__call__(*inp, **kwargs)

    def __init__(
            self,
            state_dim: int = 32,
            hidden_dim: int = 256,
            action_dim: int = 3,
            actor_opt_lr: float = 1e-3,
            critic_opt_lr: float = 1e-3,
            mem_len: int = 10**5,
            noise_scale: float = 0.1,
            soft_update_scale: float = 0.001,
            device: str = "cpu"
    ):
        super().__init__()
        self.soft_update_scale = soft_update_scale
        self.device = torch.device(device)
        self.memory = deque(maxlen=mem_len)

        self.actor = Actor(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, device=device)
        self.tactor = Actor(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, device=device)
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=actor_opt_lr)

        self.critic = Critic(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, device=device)
        self.tcritic = Critic(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, device=device)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=critic_opt_lr)

        self.hard_update()

        self.noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(action_dim), sigma=noise_scale * np.ones(action_dim))
        self.to(self.device)

    def reset_noise(self):
        self.noise.reset()

    def soft_update(self):
        def supdate(target: nn.Module, source: nn.Module, tau):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
        supdate(self.tactor, self.actor, self.soft_update_scale)
        supdate(self.tcritic, self.critic, self.soft_update_scale)

    def hard_update(self):
        self.tactor.load_state_dict(self.actor.state_dict())
        self.tcritic.load_state_dict(self.critic.state_dict())

    def memadd(self, t):
        (z, h), a, r, (nz, nh) = t
        h, nh = h[0].squeeze(), nh[0].squeeze()
        t = torch.cat((z, h), dim=-1), a, r, torch.cat((nz, nh), dim=-1)
        self.memory.append(t)

    def optimize(self, batch_size: int = 64):
        batch = random.sample(self.memory, batch_size)
        state = torch.stack([i[0] for i in batch]).detach().to(self.device)
        a = torch.stack([torch.from_numpy(i[1]) for i in batch]).float().detach().to(self.device)
        rew = torch.stack(
            [torch.from_numpy(np.array(i[2])).unsqueeze(dim=0) for i in batch]
        ).float().detach().to(self.device)
        next_state = torch.stack([i[3] for i in batch]).detach().to(self.device)

        # optimize critic
        with torch.no_grad():
            next_a = self.tactor.forward(next_state).detach()
            next_val = self.tcritic.forward(next_state, next_a).detach().squeeze()
            y_expected = rew.squeeze() + 0.99 * next_val

        y_pred = self.critic.forward(state, a).squeeze()
        loss_critic = f.smooth_l1_loss(y_pred, y_expected)
        self.copt.zero_grad()
        loss_critic.backward()
        self.copt.step()

        # optimize actor
        pred_a = self.actor.forward(state)
        loss_actor = -1 * torch.sum(self.critic.forward(state, pred_a))
        self.aopt.zero_grad()
        loss_actor.backward()
        self.aopt.step()

        self.soft_update()

    def play_act(self, z_state: torch.Tensor, h_state: Tuple[torch.Tensor, torch.Tensor], explore: bool = True):
        with torch.no_grad():
            h_state = h_state[0].squeeze()
            actions = self.actor.forward(torch.cat((z_state, h_state), dim=-1)).detach().squeeze().numpy()
            if explore:
                actions += self.noise()
            return actions

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, *args, **kwargs):
        state_dict = torch.load(path)
        cnt = Controller(*args, **kwargs)
        cnt.load_state_dict(state_dict=state_dict)
        return cnt
