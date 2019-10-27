import pickle

import gym
from tqdm import tqdm
from worldmodels.VAE import VAE
from worldmodels.envwrappers import PreprocessCarRacing
from torchvision.transforms.functional import to_tensor

env = gym.make("CarRacing-v0")
env = PreprocessCarRacing(env)

vae = VAE.load_model("binary_data/vae.pkl")
vae.eval()

games_number = 10
games = []
current_game = []

for i in tqdm(range(games_number)):
    obs = env.reset()
    while True:
        z = vae.reparameterize(*vae.encode(to_tensor(obs).float().unsqueeze(0)))
        act = env.action_space.sample()
        next_obs, _, done, _ = env.step(act)

        next_z = vae.reparameterize(*vae.encode(to_tensor(next_obs).float().unsqueeze(0)))
        current_game.append((z, act, next_z))
        obs = next_obs

        if done:
            games.append(current_game)
            break

with open("binary_data/2.2.games_for_model.pkl", 'wb') as ff:
    pickle.dump(games, ff)