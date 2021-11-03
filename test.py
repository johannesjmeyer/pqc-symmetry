import numpy as np
import tictactoe as ttt

game = np.array([[-1, 1, 1], [0, -1, 1], [-1, 0, -1]])

print(ttt.check_game(game))