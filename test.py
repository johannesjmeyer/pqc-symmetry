import numpy as np
import tictactoe as ttt
import matplotlib.pyplot as plt

game = np.array([[-1, 1, 1], [0, -1, 1], [-1, 0, -1]])
game2 = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 0]])

games, labels = ttt.get_data()

fig = plt.figure()
ttt.plot(game, plt.gca())
plt.show()