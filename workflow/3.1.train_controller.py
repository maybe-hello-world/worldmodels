import gc
import gym
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from utils.torch_extensions import get_model_params_size
from worldmodels.envwrappers import PreprocessCarRacing
from worldmodels.VAE import VAE
from worldmodels.model import MDNRNN
from worldmodels.controller import Controller

import torch
torch.set_num_threads(1)


def rollout(
        vae_: VAE,
        model_: MDNRNN,
        controller_: Controller,
        env_: gym.Env,
        length: int = 1000,
        batch_size: int = 64,
        optimize: bool = True
):
    length = length
    obs = env_.reset()
    controller.reset_noise()
    h = model_.init_state()
    done = False
    cumulative_reward = 0
    z = vae_.play_encode(obs)    # shape: (32,)
    for _ in tqdm(range(length)):
        if done:
            break
        # get action and play
        a = controller_.play_act(z, h)

        obs, rew, done, _ = env_.step(a)
        cumulative_reward += rew

        # get next states for model and vae
        n_h = model_.play_predict(a, z, h)
        n_z = vae_.play_encode(obs)

        # add to memory
        controller_.memadd(((z, h), a, rew, (n_z, n_h)))
        z, h = n_z, n_h

        if optimize:
            controller_.optimize(batch_size=batch_size)
    gc.collect()
    return cumulative_reward


sns.set()

device = "cpu"

env = gym.make("CarRacing-v0")
env = PreprocessCarRacing(env)

vae = VAE.load_model("binary_data/vae.pkl", device=device)
vae.eval()

model = MDNRNN.load_model("binary_data/model.pkl", device=device)
model.eval()

controller = Controller(device=device)
controller.train()

params, size = get_model_params_size(controller)
print(f"Params: {params}, approximately {size} Mb")

episode_length = 500
games = 5
batch_size = 16

rews = []

# fill buffer
rollout(vae, model, controller, env, length=batch_size * 5, optimize=False)

# train
for i in range(games):
    print(f"Epoch {i+1}/{games}")
    cur_rew = rollout(vae, model, controller, env, length=episode_length, batch_size=batch_size)
    rews.append(cur_rew)
    print(cur_rew)

controller.save_model("binary_data/controller.pkl")

sns.lineplot(range(len(rews)), rews)
plt.show()
