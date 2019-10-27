import pickle
from tqdm import tqdm
from worldmodels.VAE import VAE
from torchvision.transforms.functional import to_tensor, to_pil_image
import matplotlib.pyplot as plt
import random


vae = VAE.load_model("binary_data/vae.pkl", z_dim=32, image_height=42, image_width=48)
frames = 3
with open("binary_data/1.frames_carracing.pkl", 'rb') as f:
    obses = pickle.load(f)

vae.train(mode=False)
for i in tqdm(range(frames)):
    obs = random.sample(obses, 1)[0]
    nobs = to_tensor(obs).float().unsqueeze(0)
    nobs, _, _ = vae.forward(nobs)
    nobs = nobs.detach().squeeze()
    nobs = to_pil_image(nobs, mode='RGB')
    plt.imshow(obs.squeeze())
    plt.show()
    plt.imshow(nobs)
    plt.show()