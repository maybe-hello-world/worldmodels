import gym
import torch
from worldmodels.envwrappers import PreprocessCarRacing
from worldmodels.VAE import VAE
from worldmodels.model import MDNRNN
from torchvision.transforms.functional import to_tensor, to_pil_image
from matplotlib import pyplot as plt

env = gym.make("CarRacing-v0")
env = PreprocessCarRacing(env)

vae = VAE.load_model("binary_data/vae.pkl")
vae.eval()

model = MDNRNN.load_model("binary_data/model.pkl")
model.eval()

done = True
show = False


while True:
    if done:
        hidden = (
            torch.zeros((1, 1, 256)),
            torch.zeros((1, 1, 256))
        )
        obs = env.reset()

        # skip first 10 steps (beginning of the game)
        for i in range(20):
            env.step(env.action_space.sample())

    env.render()
    obs = to_tensor(obs).float().unsqueeze(0)  # 1x3x56x64
    obs = vae.reparameterize(*vae.encode(obs)).unsqueeze(0)  # 1x1x32

    actions = torch.from_numpy(env.action_space.sample())  # 3

    mus, _, pis, hidden = model.forward(actions.unsqueeze(0).unsqueeze(0), obs, hidden)
    next_obs = model.get_eval(mus, pis)
    next_obs = to_pil_image(vae.decode(next_obs).detach().squeeze())

    if show:
        plt.imshow(next_obs)
        plt.show()

    next_obs, _, done, _ = env.step(actions.numpy())
    if show:
        plt.imshow(next_obs)
        plt.show()

    obs = next_obs







