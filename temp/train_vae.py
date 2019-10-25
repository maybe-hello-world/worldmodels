import seaborn as sns
from worldmodels.VAE import VAE
import pickle
from matplotlib import pyplot as plt
import numpy as np


sns.set()

with open('frames.pkl', 'rb') as f:
    frames = pickle.load(f)

vae = VAE()

model_parameters = filter(lambda p: p.requires_grad, vae.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

losses = vae.train_model(frames, epochs=1, batch_size=64)

sns.lineplot(x=range(len(losses)), y=losses)
plt.show()