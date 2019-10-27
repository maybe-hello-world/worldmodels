"""
Should be needed if sequences are not equal-sized.
TODO: maybe do it more clever?
"""

import pickle
import sys

with open("binary_data/2.games_for_model.pkl", 'rb') as ff:
    games = pickle.load(ff)

min_len = min(len(x) for x in games)
mean_len = sum(len(x) for x in games) / len(games)

ans = input(f"min length: {min_len}, mean length: {mean_len}, continue? [N/y]")
if ans.lower() != "y" or sys.argv[1] == "y":
    print("Exiting without any data modifications...")
    exit(0)

print("Modificating data...")
games = [x[:min_len] for x in games]

with open("binary_data/2.games_for_model.pkl", "wb") as ff:
    pickle.dump(games, ff)
