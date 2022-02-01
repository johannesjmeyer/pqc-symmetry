from pennylane import numpy as np
import itertools
import matplotlib.pyplot as plt
from sqlalchemy import all_
import random
from tqdm import tqdm

MAGIC_SQUARE = np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])

def check_game(game):
    M = MAGIC_SQUARE * game
    sums = np.array([*np.sum(M, axis=0), *np.sum(M, axis=1), np.trace(M), np.trace(np.flipud(M))])

    # (O wins, X wins)
    return (15 in sums, -15 in sums)

def get_label(game):
    checks = check_game(game)

    if np.sum(game) not in [0, -1]:
        return None

    if checks[0] and checks[1]:
        return None
    
    if checks[0]:
        return 1

    if checks[1]:
        return -1

    return 0


def is_valid(game):
    checks = check_game(game)

    return (checks[0] and checks[1]) is not True

def make_data():
    all_games = []
    all_labels = []

    all_arrays = itertools.product([-1, 0, 1], repeat=9)

    for array in all_arrays:
        game = np.array(array).reshape((3,3))
        label = get_label(game)

        if label is None:
            continue

        all_games.append(game)
        all_labels.append(label)

    all_games = np.array(all_games)
    all_labels = np.array(all_labels)

    return all_games, all_labels

def get_data():
    try:
        print("Loading games and labels")
        all_games = np.load("tictactoe_games.npy")
        all_labels = np.load("tictactoe_labels.npy")
    except IOError:
        print("No file found, creating data")
        all_games, all_labels = make_data()

        print("Saving data for future use")
        np.save("tictactoe_games.npy", all_games)
        np.save("tictactoe_labels.npy", all_labels)

    print("Data loading complete")

    return all_games, all_labels

def get_data_symmetric():
    try:
        print("Loading games and labels but excluding equivalent games under symmetry")
        all_games = np.load("tictactoe_games_symmetric.npy")
        all_labels = np.load("tictactoe_labels_symmetric.npy")
    except IOError:
        print("No file found, creating data")
        all_games, all_labels = make_symmetric(*get_data())

        print("Saving data for future use")
        np.save("tictactoe_games_symmetric.npy", all_games)
        np.save("tictactoe_labels_symmetric.npy", all_labels)

    print("Data loading complete")

    return all_games, all_labels

def make_symmetric(games, labels = None):
    """
    Creates new dataset excluding games that are symetrically equivalent (rotation, mirroring).
    This is terribly inefficient but we only need to run this once
    """
    symmetric_games = []
    symmetric_labels = []
    for i, game in tqdm(enumerate(games)):
        symmetries = get_symmetries(game)
        
        in_array = False
        for g in symmetries:
            if any((g==x).all() for x in symmetric_games):
                in_array = True
                break
        if not in_array:
            symmetric_games.append(random.choice(symmetries))
            if labels is None:
                symmetric_labels.append(get_label(game))
            else:
                symmetric_labels.append(labels[i])

    return symmetric_games, symmetric_labels

def get_symmetries(game):

    all_symmetries = [game]      

    for i in range(3):
        all_symmetries.append(np.rot90(all_symmetries[-1]))

    all_symmetries.append(np.flipud(game))
    all_symmetries.append(np.fliplr(game))

    return all_symmetries

def plot(game, ax):
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axvline(1, color="gray")
    ax.axvline(2, color="gray")
    ax.axhline(1, color="gray")
    ax.axhline(2, color="gray")

    for idx in np.ndindex(*game.shape):
        val = game[idx]
        if val == 1:
            # ax.scatter(idx[0] + .5, idx[1] + .5, marker="o", lw=8, s=4000, facecolors="none", edgecolors="C1")                    
            ax.add_patch(plt.Circle((idx[0] + .5, idx[1] + .5), 0.3, color='C1', lw=8, fill=False))
        elif val == -1:
            D = 0.27
            ax.plot([idx[0] + .5 - D, idx[0] + .5 + D], [idx[1] + .5 - D, idx[1] + .5 + D], lw=8, color="C0")
            ax.plot([idx[0] + .5 - D, idx[0] + .5 + D], [idx[1] + .5 + D, idx[1] + .5 - D], lw=8, color="C0")

    ax.axis("off")
# %%
