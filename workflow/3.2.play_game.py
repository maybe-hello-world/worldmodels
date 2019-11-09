import gym

from worldmodels.envwrappers import PreprocessCarRacing
from worldmodels.VAE import VAE
from worldmodels.model import MDNRNN
from worldmodels.controller import Controller

import matplotlib.pyplot as plt
import seaborn as sns


def rollout(
        vae_: VAE,
        model_: MDNRNN,
        controller_: Controller,
        env_: gym.Env,
        length: int = 10000
):
    obs = env_.reset()
    h = model_.init_state()
    done = False
    cumulative_reward = 0
    z = vae_.play_encode(obs)    # shape: (32,)
    for _ in range(length):
        if done:
            break
        env.render()
        # get action and play
        a = controller_.play_act(z, h, explore=False)
        obs, rew_, done, _ = env_.step(a)
        cumulative_reward += rew_

        # get next states for model and vae
        h = model_.play_predict(a, z, h)
        z = vae_.play_encode(obs)
    return cumulative_reward


sns.set()
device = "cpu"
games = 5

env = gym.make("CarRacing-v0")
env = PreprocessCarRacing(env)

vae = VAE.load_model("binary_data/vae.pkl", device=device)
vae.eval()

model = MDNRNN.load_model("binary_data/model.pkl", device=device)
model.eval()

controller = Controller.load_model("binary_data/controller.pkl", device=device)
controller.eval()

rews = []
for i in range(games):
    rew = rollout(vae, model, controller, env, length=200)
    rews.append(rew)

sns.lineplot(range(len(rews)), rews)
plt.show()
