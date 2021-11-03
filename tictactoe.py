import numpy as np
import itertools

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
