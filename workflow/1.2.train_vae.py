import seaborn as sns
from worldmodels.VAE import VAE
import pickle
from matplotlib import pyplot as plt
from utils.torch_extensions import get_model_params_size


sns.set()

with open('binary_data/1.frames_carracing.pkl', 'rb') as f:
    frames = pickle.load(f)

vae = VAE()

params, size = get_model_params_size(vae)
print(f"Params: {params}, approximately {size} Mb")

# TODO: add horizontal flips
vae_l = vae.train_model(frames, epochs=5, batch_size=64)
vae.save_model("binary_data/vae.pkl")

sns.lineplot(x=range(len(vae_l)), y=vae_l)
plt.show()
