# %%
from pennylane import numpy as np
import itertools
import matplotlib.pyplot as plt
from sqlalchemy import all_
import random
from tqdm import tqdm
import tictactoe as tic

"""
def make_data():
    all_roads = []
    all_labels = []

    all_roads += tic.get_symmetries(np.array([[-1, -1, -1], [1, 1, 1], [-1, -1, -1]], dtype=float))
    all_roads += tic.get_symmetries(np.array([[1, 1, 1], [-1, 1, -1], [-1, 1, -1]], dtype=float))
    all_roads += [np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]], dtype=float)]
    all_roads += tic.get_symmetries(np.array([[-1, -1, -1], [1, 1, 1], [-1, 1, -1]], dtype=float))
    all_roads += tic.get_symmetries(np.array([[1, 1, 1], [1, -1, -1], [1, -1, -1]], dtype=float))
    all_roads += tic.get_symmetries(np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, -1]], dtype=float))

    all_situations = []
    for road in all_roads:
        surface = np.argwhere(road>0)
        for i in surface:
            for k in surface:
                if np.linalg.norm(i-k) == 1.:
                    situation = np.copy(road)
                    situation[i[0], i[1]] = 0
                    situation[k[0], k[1]] = 0.5
                    all_situations.append(situation)
                    all_labels.append(get_label(situation))    

    return all_situations, all_labels
"""   
def get_symmetries(game):

    all_symmetries = [game]      

    for i in range(3):
        all_symmetries.append(np.rot90(all_symmetries[-1]))

    all_symmetries.append(np.flipud(game))
    all_symmetries.append(np.fliplr(game))

    # add diagnoal symmetry
    all_symmetries.append(game.T)
    all_symmetries.append(np.fliplr((np.fliplr(game)).T))

    return all_symmetries

shapes = []
shapes += [np.array([[1, 1, 1], [-1, -1, -1], [-1, -1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, -1, -1], [1, 1, 1], [-1, -1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[1, 1, 1], [-1, 1, -1], [-1, 1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[1, 1, 1], [1, -1, 1], [1, 1, 1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, -1, -1], [1, 1, 1], [-1, 1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[1, 1, 1], [1, -1, -1], [1, -1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, 1, 1], [-1, 1, -1], [-1, 1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, 1, -1], [1, 1, 1], [-1, 1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[1, -1, 1], [1, 1, 1], [1, -1, 1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, -1, 1], [1, 1, 1], [1, -1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, -1, -1], [1, 1, 1], [1, -1, 1]], dtype=float, requires_grad = False)]
shapes += [np.array([[1, -1, -1], [1, 1, 1], [1, -1, 1]], dtype=float, requires_grad = False)]
shapes += [np.array([[1, 1, 1], [1, -1, 1], [1, -1, 1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, 1, -1], [1, 1, -1], [1, -1, -1]], dtype=float, requires_grad = False)]
shapes += [np.array([[-1, 1, -1], [1, 1, 1], [1, -1, -1]], dtype=float, requires_grad = False)]

def make_data():
    all_roads = []
    all_labels = []

    for shape in shapes:
        all_roads += tic.get_symmetries(shape)

    all_roads = list(np.unique(np.array(all_roads), axis=0))

    all_situations = []
    for road in all_roads:
        surface = np.argwhere(road>0)
        for i in surface:
            situation = np.copy(road)
            situation[i[0], i[1]] = 0
            directions = get_directions(situation)
            for direction in directions:
                all_situations.append([situation, direction])
                all_labels.append(get_label([situation, direction]))    

    return all_situations, all_labels
"""
def get_label(road):
    n = road.shape[0]
    pos = (np.argwhere(road == 0.5))[0]
    forward = (np.argwhere(road == 0.5) - np.argwhere(road == 0))[0]
    right = pos + np.array([forward[1], -forward[0]])
    left = pos + np.array([-forward[1], forward[0]])
    forward += pos
    label = []

    if not np.any(forward<0) and not np.any(forward>=n):
        if road[forward[0], forward[1]] == 1:
            label.append('f')

    if not np.any(right<0) and not np.any(right>=n):
        if road[right[0], right[1]] == 1:
            label.append('r')

    if not np.any(left<0) and not np.any(left>=n):
        if road[left[0], left[1]] == 1:
            label.append('l')

    if not label:
        label.append('s')
    
    return label
"""   

def get_label(situation):
    road = situation[0]
    direction = situation[1]
    n = road.shape[0]

    if direction == 'n':
        forward = np.array([-1, 0])
    elif direction == 's':
        forward = np.array([1, 0])
    elif direction == 'w':
        forward = np.array([0, -1])
    elif direction == 'e':
        forward = np.array([0, 1])
    else:
        print('error')
        return ['s']

    pos = np.argwhere(road == 0)[0] + forward

    right = pos + np.array([forward[1], -forward[0]])
    left = pos + np.array([-forward[1], forward[0]])
    forward += pos
    label = []

    if not np.any(forward<0) and not np.any(forward>=n):
        if road[forward[0], forward[1]] == 1:
            label.append('f')

    if not np.any(right<0) and not np.any(right>=n):
        if road[right[0], right[1]] == 1:
            label.append('r')

    if not np.any(left<0) and not np.any(left>=n):
        if road[left[0], left[1]] == 1:
            label.append('l')

    if not label:
        label.append('s')
    
    return label

def get_directions(road):
    n = road.shape[0]
    pos = (np.argwhere(road == 0))[0]
    north = pos + np.array([-1, 0])
    south = pos + np.array([1, 0])
    west = pos + np.array([0, -1])
    east = pos + np.array([0, 1])
    label = []

    if not np.any(north<0) and not np.any(north>=n):
        if road[north[0], north[1]] == 1:
            label.append('n')

    if not np.any(south<0) and not np.any(south>=n):
        if road[south[0], south[1]] == 1:
            label.append('s')

    if not np.any(west<0) and not np.any(west>=n):
        if road[west[0], west[1]] == 1:
            label.append('w')

    if not np.any(east<0) and not np.any(east>=n):
        if road[east[0], east[1]] == 1:
            label.append('e')

    if not label:
        label.append('x')

    return label

def get_diff(situation):
    label = get_label(situation)
    if label == ['f'] or label == ['s']:
        return -1
    elif label == ['r'] or label == ['l']:
        return -0.6
    elif label == ['f', 'r']:
        return -0.2
    elif label == ['f', 'l']:
        return 0.2
    elif label == ['r', 'l']:
        return 0.6
    elif label == ['f', 'r', 'l']:
        return 1
    else:
        raise AttributeError()

def gen_road(n, l):
    road = np.ones((n, n))*-1
    prev_step = random.choice(get_border(n))
    road[prev_step[0], prev_step[1]] = 1
    neighbors = [[0, 1],[0, -1],[1, 0],[-1, 0]]
    for _ in range(l):
        direction = [prev_step + np.array(i) for i in neighbors if not np.any((prev_step + np.array(i))<0) and not np.any((prev_step + np.array(i))>=n)]
        step = random.choice(direction)
        road[step[0], step[1]] = 1
        prev_step = step

    return road

def get_border(n):
    return [[0, i] for i in range(n)] + [[n-1, i] for i in range(n)] + [[i, 0] for i in range(1, n-1)] + [[i, n-1] for i in range(1, n-1)]



# %%
