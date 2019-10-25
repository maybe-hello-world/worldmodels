import gym
import pickle
from tqdm import tqdm

env = gym.make("Breakout-v0")

obses = []
games = 20
for i in tqdm(range(games)):
    obs = env.reset()
    while True:
        obses.append(obs)
        act = env.action_space.sample()
        obs, rew, done, _ = env.step(act)
        if done:
            break

env.close()

with open('frames.pkl', 'wb') as f:
    pickle.dump(obses, f)
