import pickle
from matplotlib import pyplot as plt
from worldmodels.model import MDNRNN
from utils.torch_extensions import get_model_params_size
import seaborn as sns
sns.set()

with open("binary_data/2.games_for_model.pkl", "rb") as ff:
    games = pickle.load(ff)

model = MDNRNN()
params, size = get_model_params_size(model)
print(f"Params: {params}, approximately {size} Mb")

model_loss = model.train_model(games, epochs=3)
model.save_model("binary_data/model.pkl")

sns.lineplot(x=range(len(model_loss)), y=model_loss)
plt.show()
