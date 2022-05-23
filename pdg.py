# %%
from cmath import pi
from http.client import responses
import pennylane as qml
from pennylane import numpy as np
from tictactoe import *
import random
from copy import copy, deepcopy
from tabulate import tabulate
from timeit import default_timer as timer 
import os
from sklearn.metrics import confusion_matrix
from torch import multiprocessing as mp
from torch.multiprocessing import Pool
import torch

import circuits_ttt as pqc
from intersection import *

if __name__ == "__main__":
    args_symmetric = {'c': 2, 'e': 2, 'o': 2, 'm': 2, 'i': 1, 'd': 1, 'z': 3, 'x':2}
    args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4, 'z': 9, 'x':2}
    gate_2q = qml.CRX
    rotation_2q = True

def data_encoding(intersection, direction):
    fintersection = intersection.flatten()
    order = [0, 1, 2, 5, 8, 7, 6, 3, 4]

    for i, j in enumerate(order):
        qml.RX(fintersection[j], wires=i)
    
    if direction == 'n':
        qml.RX(0, wires = 9)
    elif direction == 'e':
        qml.RX(np.pi/2, wires = 9)
    elif direction == 's':
        qml.RX(np.pi, wires = 9)
    elif direction == 'w':
        qml.RX(np.pi*(3/2), wires = 9)
    """
    if direction == ['w']:
        qml.RX(np.pi/2, wires = 9)
    elif direction == ['e']:
        qml.RX(-np.pi/2, wires = 9)
    elif direction == ['n']:
        qml.RY(np.pi/2, wires = 9)
    elif direction == ['s']:
        qml.RY(-np.pi/2, wires = 9)
    """
    

def entangle_direction(param, symm=True):
    
    edges = [1, 3, 5, 7]
    corners = [0, 2, 4, 6]
    middle = [8]
    group = [edges, middle, corners]

    if symm:
        for i, symmetry in enumerate(group):
            for k in symmetry:
                gate_2q(param[i], wires=[k, 9])
    else:
        for i in range(9):
                gate_2q(param[i], wires=[i, 9])

def sq_direction(param):

    qml.RX(param[0], wires=9)
    qml.RY(param[1], wires=9)

### Symmetric functions ###
def corners(param, symm=True):
    '''
    applies RX an RZ gates on corners
    '''
    qubits = [0, 2, 4, 6]

    if symm:
        for i in qubits:
            qml.RX(param[0], wires=i)
            qml.RY(param[1], wires=i)

    else:
        for n, i in enumerate(qubits):
            qml.RX(param[i], wires=i)
            qml.RY(param[i+1], wires=i)

def edges(param, symm=True):
    '''
    applies RX an RZ gates on edges
    '''
    qubits = [1, 3, 5, 7]

    if symm:
        for i in qubits:

            qml.RX(param[0], wires=i)
            qml.RY(param[1], wires=i)

    else:
        for n, i in enumerate(qubits):
            qml.RX(param[i-1], wires=i)
            qml.RY(param[i], wires=i)
    
    
def center(param):
    '''
    applies RX an RZ gate on the middle qubit
    '''
    qml.RX(param[0], wires=8)
    qml.RY(param[1], wires=8)

def outer_layer(param, symm=True):
    '''
    entangles nearest neighboar qubits of the outer layer of the tictactoe board
    0  - 1 -  2
    |         |
    7    8    3
    |         |
    6  - 5 -  4
    '''
    """"""
    corners = [0, 2, 4, 6]
    edges = [1, 3, 5, 7]
    if rotation_2q:
        if symm:
            for i in range(4):
                    gate_2q(param[0], wires=[corners[i], edges[i]])
                    gate_2q(param[1], wires=[corners[i], edges[i-1]])
        else:
            for i in range(4):
                    gate_2q(param[2*i], wires=[corners[i], edges[i]])
                    gate_2q(param[2*i+1], wires=[corners[i], edges[i-1]])
    else:
        for i in range(4):
            gate_2q(wires=[corners[i], edges[i]])
            gate_2q(wires=[corners[i], edges[i-1]])      

def inner_layer(param, symm=True):
    '''
    entangles center qubit with edge qubits
    0    1    2
         |
    7  - 8 -  3
         |
    6    5    4
    '''
    edges = [1, 3, 5, 7]
    if rotation_2q:
        if symm:
            for i in edges:
                gate_2q(param[0], wires=[i, 8])
        else:
            for n, i in enumerate(edges):
                gate_2q(param[n], wires=[i, 8])
    else:
        for i in edges:
            gate_2q(wires=[i, 8]) 
  
def diag_layer(param, symm=True):
    '''
    entangles center qubit with corner qubits
    0    1    2 
      \     /
    7    8    3
      /     \ 
    6    5    4
    '''
    corners = [0, 2, 4, 6]
    if rotation_2q:
        if symm:
            for i in corners:
                gate_2q(param[0], wires=[8, i])
        else:
            for n, i in enumerate(corners):
                gate_2q(param[n], wires=[8, i])
    else:
        for i in corners:
            gate_2q(wires=[8, i])

############################################################

def cost_function_batch_old(circ, params, games, symmetric, design):

    final_results = torch.zeros(len(games))
    for i, g in enumerate(games):

        result = circ(g,params, symmetric, design=design)#.detach().numpy()
        label = get_label(g)
        
        won = torch.zeros(3)

        if 'l' in label:
            won[0] = 1

        if 'r' in label:
            won[1] = 1

        if 's' in label or 'f' in label:
            won[2] = 1
            
        avg_result = get_results(result)
        final_results[i] = torch.sum((avg_result - won)**2)
    
    return torch.mean(final_results)

def get_results(result):
    avg_result = []
    slicer = [[0, 2, 4, 6], [8], [1, 3, 5, 7]]
    for i in slicer:
        avg_result.append(torch.mean(result[i]).reshape(1))

    result = (torch.cat(avg_result)+1)/2
    return result

def cost_function_batch(circ, params, games, symmetric, design):

    loss = 0
    for i, g in enumerate(games):

        result = circ(g,params, symmetric, design=design)
        difficulty = get_diff(g)
            
        loss += (result - difficulty)**2
    
    return loss/(len(games))

def translate_to_parameters(design, symmetric=True):
    '''
    Translates circuit design string to number of required parameters in the form of a list corresponding to each applied layer
    '''
    param_args = [0]
    if symmetric:
        args = args_symmetric
    else:
        args = args_asymmetric

    for i in design.replace(" ", ""):
        if i == 't':
            # encode data
            param_args.append(param_args[-1])
        elif i == 'c':
            # corners
            param_args.append(param_args[-1]+args['c'])
        elif i == 'e':
            # edges
            param_args.append(param_args[-1]+args['e'])
        elif i == 'o':
            # outer layer
            param_args.append(param_args[-1]+args['o'])
        elif i == 'm':
            # center
            param_args.append(param_args[-1]+args['m'])
        elif i == 'i':
            # inner layer
            param_args.append(param_args[-1]+args['i'])
        elif i == 'd':
            # diagonal layer
            param_args.append(param_args[-1]+args['d'])
        elif i == 'z':
            # entangle direction
            param_args.append(param_args[-1]+args['z'])
        elif i == 'x':
            # direction
            param_args.append(param_args[-1]+args['x'])
        
    return param_args

def circuit(situation, params, symmetric, design="tceocem tceicem tcedcem"):
    '''
    prepares the all zero comp basis state then iterates through encoding and layers
    encoding and layers are defined by design argument
        game:   game as 3x3 arraa
        params: tensor of parameters
        symmetric: bool
        design: string specifying order of encoding layers
                t: encode game
                c: corners
                e: edges
                m: middle/center
                o: outer layer
                i: inner layer
                d: diagonal layer
    '''
    # TODO: this should automatically start from the all-zero state in comp basis right?

    args = translate_to_parameters(design, symmetric)

    road = situation[0].copy()

    if True:
        pos = (np.argwhere(road == 0))[0]
        north = pos + np.array([-1, 0])
        south = pos + np.array([1, 0])
        west = pos + np.array([0, -1])
        east = pos + np.array([0, 1])

        road[pos[0], pos[1]] = -1/3
        if situation[1] == 'n':
            road[north[0], north[1]] = 1/3
        elif situation[1] == 's':
            road[south[0], south[1]] = 1/3
        elif situation[1] == 'w':
            road[west[0], west[1]] = 1/3
        elif situation[1] == 'e':
            road[east[0], east[1]] = 1/3
        
        direction = 'x'
    else:
        direction = situation[1]

    ngame = np.pi*road*(2/3) # normalize entries of game so they are between -pi/2, pi/2

    for r in range(params.shape[0]): # r repetitions

        for n, i in enumerate(design.replace(" ", "")):
            if i == 't':
                data_encoding(ngame, direction)
            elif i == 'c':
                corners(params[r, args[n]:args[n+1]], symmetric)
            elif i == 'e':
                edges(params[r, args[n]:args[n+1]], symmetric)
            elif i == 'o':
                outer_layer(params[r, args[n]:args[n+1]], symmetric) 
            elif i == 'm':
                center(params[r, args[n]:args[n+1]])
            elif i == 'i':
                inner_layer(params[r, args[n]:args[n+1]], symmetric) 
            elif i == 'd':
                diag_layer(params[r, args[n]:args[n+1]], symmetric)
            elif i == 'z':
                entangle_direction(params[r, args[n]:args[n+1]], symmetric)
            elif i == 'x':
                sq_direction(params[r, args[n]:args[n+1]])

    return qml.expval(qml.PauliZ(8))
    #return [qml.expval(qml.PauliZ(i)) for i in range(9)]

games_data,labels = make_data()

games_data = [games_data[i] for i in range(len(games_data)) if not labels[i] == ['s']]

def fill_list(ls, size):
    if len(ls) == size:
        return
    elif len(ls) > size:
        raise ValueError('list already larger than requested size')
    
    ls += (size%len(ls)-1)*ls
    while len(ls) < size:
        ls += [random.choice(ls)]

def gen_games_sample(size, output = None):
    '''
    Generates Tensor with 3*size games that are won equally by X, O and 0
    If the parameter output is a string instead of "None", the sample is stored in a npz file named after the string

    param wins: list of wins to be included in sample. If empty, returns completely random sample.
    '''
    sample = []
    for k in [-1, -0.6, -0.2, 0.2, 0.6, 1]:
        sample_temp = [i for i in games_data if get_diff(i) == k]
        try:
            sample += random.sample(sample_temp, size)
        except ValueError:
            print(f'Not enough data points for difficulty {k} \nNumber of data points: {len(sample_temp)}')
            fill_list(sample_temp, size)
            sample += sample_temp

    sample_difficulty = [get_diff(i) for i in sample]

    if not output == None:
        with open('samples_pdg/'+output+'.npy', 'wb') as f:
                np.save(f,np.tensor(sample, requires_grad=False))#, sample_label = sample_label)


    return sample, np.tensor(sample_difficulty, requires_grad=False)


def random_params(repetitions, symmetric, design):
    '''
    returns array/torch tensor of paramters for amount of repetitions and circuit design
    '''
    params = rng.uniform(low=-1, high=1, size=(repetitions,translate_to_parameters(design, symmetric)[-1]))*np.pi

    return torch.tensor(params, requires_grad = True)

pdg_dev = qml.device("lightning.qubit", wires=10)
full_circ_torch = qml.QNode(circuit, pdg_dev, interface='torch', diff_method='adjoint')
rng = np.random.default_rng()

class road_detection():

    def __init__(self, symmetric=True, sample_size=5, design="tcemoidcemoidtcmoid", data_file=None, random_sample=False):
        #self.opt = qml.GradientDescentOptimizer(0.01)
        self.sample_size = sample_size
        self.design = design
        self.random = random_sample

        self.cost_function = cost_function_batch
            #self.cost_function_torch = cost_function_batch_torch

        if data_file == None:
            self.sample_games(sample_size)
        else:
            self.load_games(data_file, sample_size) # loads games and labels from file

        self.symmetric = symmetric

    def random_parameters(self, repetitions=2):
        '''
        sets random parameters for circuit design and amount of repetitions
        '''
        self.repetitions = repetitions
        self.init_params_torch = random_params(repetitions, self.symmetric, self.design)
        #self.init_params = random_params(repetitions, self.symmetric, self.design)

    def sample_games(self, size):
        '''
        Create random samples with equal amount of wins for X, O and 0
        ''' 
        self.games_sample , self.label_sample = gen_games_sample(size)

    def load_games(self, data_file, size):
        '''
        Loads games and label from file specified by data_file. The first size data points are retained.
        If the file is not found, generate a new sample.
        Currently, load is implemented via numpy.load(file.npz)
        '''
        try:
            with open('samples_pdg/'+data_file+'.npy', 'rb') as f:
                            print('Loading data file \n')
                            self.games_sample = [[i[0], str(i[1])] for i in np.tensor(np.load(f, allow_pickle=True), requires_grad=False)]
                            #self.label_sample = np.tensor(np.load(f, allow_pickle = True)['sample_label'][:size*3], requires_grad=False)
        except IOError: 
            print('Data sample not found, creating new one')
            self.sample_games(size)

    def run_epochs(self, epochs, samplesize_per_step, steps_per_epoch, stepsize,  data_file = None):
        """
        Runs lbfgs training with different epochs
        """
        print('running epochs...')
        print(tabulate([['epochs', epochs], ['stepsize', stepsize], ['symmetric', self.symmetric], ['design', self.design], ['sample size per step', samplesize_per_step], ['steps per epoch', steps_per_epoch], \
            ['repetitions', self.repetitions]]))

        self.epochs = True

        if data_file is None:
            self.batch = gen_games_sample(int(np.ceil(steps_per_epoch*samplesize_per_step/6)))[0]
        else:
            with open('samples/'+data_file+'.npz', 'rb') as f:
                            print('Loading data file \n')
                            self.batch = np.tensor(np.load(f, allow_pickle = True)['sample'], requires_grad=False)

        self.test_batch = gen_games_sample(50)[0]

        np.random.shuffle(self.batch)
        self.batch = self.batch[:steps_per_epoch*samplesize_per_step]

        self.interface = 'torch'

        self.gd_cost = []
        self.theta = self.init_params_torch
        #self.opt = torch.optim.LBFGS([self.theta], lr=stepsize)
        #self.opt = LBFGSNew([self.theta], lr=stepsize, line_search_fn=True, batch_mode=True)
        self.opt = torch.optim.Adam([self.theta], lr=stepsize)
        self.stepsize = stepsize
        self.epoch_total_accuracy = []
        self.epoch_accuracy = []
        self.epoch_cost_function = []

        step_start = 0
        step_end = 0
        
        for i in range(epochs):

            for j, sample in enumerate([self.batch[k:k+samplesize_per_step] for k in range(0, len(self.batch), samplesize_per_step)]):
                self.games_sample = sample
                cost_temp = self.cost_function(full_circ_torch, self.opt.param_groups[0]['params'][0], self.games_sample, self.symmetric, self.design)
                print(f"epoch {i}/{epochs} step {j}/{steps_per_epoch} current cost value: {cost_temp} execution time: {step_end-step_start}s")
                step_start = timer()

                def closure():
                    self.opt.zero_grad()
                    loss = self.cost_function(full_circ_torch, self.theta, self.games_sample, self.symmetric, self.design)
                    loss.backward()
                    return loss
                
                self.opt.step(closure)
                step_end = timer()

                self.gd_cost.append(cost_temp) 
                self.steps = j
            
            np.random.shuffle(self.batch)
            #print(f'unique games: {len(np.unique(self.batch, axis=0))}')
            self.theta = self.opt.param_groups[0]['params'][0]  
            print(f'epoch {i}/{epochs} accuracy:')
            self.epoch_accuracy.append(self.check_accuracy(check_batch=self.batch))
            print(f'epoch {i}/{epochs} total accuracy:')
            self.epoch_total_accuracy.append(self.check_accuracy(check_batch=self.test_batch))
            self.epoch_cost_function.append(self.cost_function(full_circ_torch, self.opt.param_groups[0]['params'][0], self.batch, self.symmetric, self.design))
            #print(f"epoch {i}/{epochs} accuracy: {self.epoch_accuracy[-1]}")
            print(f"epoch {i}/{epochs} cost function: {self.epoch_cost_function[-1]}")
        
        print('Done')
        self.theta = self.opt.param_groups[0]['params'][0]  


    def run_lbfgs(self, steps, stepsize=0.1, resume = False):
        '''
        Runs qml with torch's lbfgs implementation. Usually converges much quicker than pennylanes standard gradient descent optimizer
        '''
        print('running lbfgs...')
        print(tabulate([['steps', steps], ['stepsize', stepsize], ['symmetric', self.symmetric], ['design', self.design], ['sample size', len(self.games_sample)], \
            ['random sample', self.random], ['repetitions', self.repetitions]]))

        self.epochs = False
        self.interface = 'torch'
        if not resume:
            self.gd_cost = []
            self.theta = self.init_params_torch
            self.total_accuracy = []
            self.training_accuracy = []

        self.test_batch = gen_games_sample(30)[0]

        self.opt = torch.optim.LBFGS([self.theta], lr=stepsize)
        self.stepsize = stepsize
        step_start = 0
        step_end = 0

        def closure():
            self.opt.zero_grad()
            loss = self.cost_function(full_circ_torch, self.theta, self.games_sample, self.symmetric, self.design)
            loss.backward()
            return loss

        self.training_accuracy.append(self.check_accuracy(check_batch=self.games_sample))
        self.total_accuracy.append(self.check_accuracy(check_batch=self.test_batch))

        for j in range(steps):

            cost_temp = self.cost_function(full_circ_torch, self.opt.param_groups[0]['params'][0],self.games_sample, self.symmetric, self.design)
            
            print(f"step {j}/{steps} current cost value: {cost_temp} execution time: {step_end-step_start}s")
            
            step_start = timer()
            self.opt.step(closure)
            step_end = timer()
            # Samples new games for every step
            if self.random:
                self.sample_games(self.sample_size)

            self.gd_cost.append(cost_temp) 
            self.theta = self.opt.param_groups[0]['params'][0]
            self.training_accuracy.append(self.check_accuracy(check_batch=self.games_sample))
            self.total_accuracy.append(self.check_accuracy(check_batch=self.test_batch))

            self.steps = j
        
        cost_temp = self.cost_function(full_circ_torch ,self.opt.param_groups[0]['params'][0],self.games_sample, self.symmetric, self.design)
        self.gd_cost.append(cost_temp)
        print(f"final step current cost value: {cost_temp}")

        self.theta = self.opt.param_groups[0]['params'][0]  
        print('accuracy on training data set:')
        #self.check_accuracy(check_batch=self.games_sample)

    
    def check_accuracy(self, check_size=30, check_batch=None):

        if check_batch is None:
            check_batch, diff_check = gen_games_sample(check_size)
        else:
            diff_check = [get_diff(i) for i in check_batch]
        results = []
        targets = []
        for i, situation in enumerate(check_batch):

            result = full_circ_torch(situation, self.theta, self.symmetric, self.design).detach().numpy()

            results.append(np.round((result+0.2)*5/2))
            targets.append(np.round((diff_check[i]+0.2)*5/2))

        self.confusion_matrix = confusion_matrix(targets, results, normalize='true')
        print('confusion matrix:')
        print(self.confusion_matrix)
        self.accuracy = self.confusion_matrix.trace()/self.confusion_matrix.shape[0]

        return self.confusion_matrix
  
    """
    def check_accuracy(self, check_size = 12, check_batch=None): # TODO: check if accuracy varies for same run
        # TODO: implement confusion matrix
        '''
        checks accuracy of current theta by sampling check_size amount of games for each win
        '''

        if check_batch is None:
            games_check = gen_games_sample(check_size)[0]
        else:
            games_check = check_batch

        labels_check = [get_label(i) for i in games_check]
        
        res_circ = []
        res_true = []

        for i, game in enumerate(games_check[:500]):

            res_true_array = ''
            if 'l' in labels_check[i]:
                res_true_array += 'l'

            if 'r' in labels_check[i]:
                res_true_array += 'r'

            if 's' in labels_check[i] or 'f' in labels_check[i]:
                res_true_array += 'f'
            
            res_true.append(res_true_array)

            res = full_circ_torch(game, self.theta, self.symmetric, design=self.design)

            avg_results = get_results(res)

            ret = ''
            if avg_results[0] > 0:
                ret += 'l'
            if avg_results[1] > 0:
                ret += 'r'
            if avg_results[2] > 0:
                ret += 'f'
            
            res_circ.append(ret)

        # check accuracy
        # confusion matrix:
        self.confusion_matrix = confusion_matrix(res_true, res_circ, normalize='true')
        self.accuracy = self.confusion_matrix.trace()/3
        print('Confusion matrix:')
        print(self.confusion_matrix)
        #self.accuracy[-1] = len([j for j in results_alt[-1] if j[0] > j[1] and j[0] > j[2]])/len(results_alt[-1])
        #self.accuracy[0] = len([j for j in results_alt[0] if j[1] > j[0] and j[1] > j[2]])/len(results_alt[0])
        #self.accuracy[1] = len([j for j in results_alt[1] if j[2] > j[1] and j[2] > j[0]])/len(results_alt[1])
    """
    def plot_cost(self):
        '''
        plots cost function
        '''
        plt.plot(self.gd_cost)
        plt.show()

    def save(self, name, exec_time=0):
        '''
        saves result of qml as a npy file. Can be analyzed later
        '''

        params_tmp = self.init_params_torch.detach().numpy()
        theta_tmp = self.theta.detach().numpy()

        to_save = {'symmetric': self.symmetric, 'total accuracy': self.total_accuracy, 'training accuracy': self.training_accuracy, 'accuracy': self.confusion_matrix,'execution time': exec_time, 'steps': self.steps, 'stepsize': self.stepsize, 'design': self.design, 'cost function': self.gd_cost, 'sample size': self.sample_size, \
        'initial parameters': params_tmp, 'sampled games': self.games_sample, 'theta': theta_tmp}
        if self.epochs:
            to_save['epoch cost'] = self.epoch_cost_function
            to_save['epoch accuracy'] = self.epoch_accuracy
            to_save['epoch batch'] = self.batch
            to_save['epoch total accuracy'] = self.epoch_total_accuracy

        #dd.io.save(name + '.h5', to_save)
        print('Saving results as {}.npy'.format(name))
        try:
            np.save(name, to_save)
        except FileNotFoundError:
            os.makedirs(os.getcwd()+'/output/'+name[::-1].split('/', 1)[1][::-1])
            np.save('output/'+name, to_save)
# %%
