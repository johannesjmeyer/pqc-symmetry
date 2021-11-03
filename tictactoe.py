import numpy as np

MAGIC_SQUARE = np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])

def check_game(game):
    M = MAGIC_SQUARE * game

    print([np.sum(M, axis=0), np.sum(M, axis=1), np.trace(M), np.trace(np.flipud(M))])
