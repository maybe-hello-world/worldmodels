from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as f


class Controller(nn.Module):
    # PyCharm problems
    # https://youtrack.jetbrains.com/issue/PY-37601
    def __call__(self, *inp, **kwargs) -> Any:
        return super().__call__(*inp, **kwargs)

    def __init__(
            self,
            z_dim_size: int = 32,
            m_hidden_size: int = 256,
            actions_size: int = 3,
            neurons: int = 128,
            device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)

        self.z_dim_size = z_dim_size
        self.m_hidden_size = m_hidden_size
        self.actions_size = actions_size
        self.neurons = neurons

        self.fc1 = nn.Linear(self.z_dim_size + self.m_hidden_size, self.neurons)
        self.fc2 = nn.Linear(self.neurons, self.actions_size)
        self.to(self.device)

    def forward(self, z_state: torch.Tensor, h_state: torch.Tensor) -> torch.Tensor:
        """
        :param z_state: z_state from VAE, shape: 1x32
        :param h_state: h_state from VAE, shape: 1x256
        :return: action
        """
        h_state = h_state[0].squeeze()  # use only hidden, throw out cell state
        x = torch.cat([z_state, h_state], dim=-1).unsqueeze(0).to(self.device)
        x = f.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def play_act(self, z_state: torch.Tensor, h_state: torch.Tensor):
        return self.forward(z_state, h_state).detach().squeeze().numpy()


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path, *args, **kwargs):
        state_dict = torch.load(path)
        vae = Controller(*args, **kwargs)
        vae.load_state_dict(state_dict=state_dict)
        return vae
