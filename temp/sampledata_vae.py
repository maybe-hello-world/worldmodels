import gym
import pickle
from tqdm import tqdm
from worldmodels.envwrappers import PreprocessCarRacing

env = gym.make("CarRacing-v0")
env = PreprocessCarRacing(env)

obses = []
frames = 10000
obs = env.reset()
for i in tqdm(range(frames)):
    obses.append(obs)
    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)
    if done:
        obs = env.reset()

env.close()

with open('binary_data/frames_carracing.pkl', 'wb') as f:
    pickle.dump(obses, f)
