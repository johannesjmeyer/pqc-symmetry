# %%
import pennylane as qml
from pennylane import numpy as np
from tictactoe import *
import random
from copy import copy, deepcopy

import torch
#import deepdish as dd
#from jax.config import config
#config.update("jax_enable_x64", True)
#import jax
#import jax.numpy as jnp

ttt_dev = qml.device("default.qubit", wires=9) # the device used to label ttt instances
# TODO: use other device https://pennylane.ai/plugins.html
# TODO: implement jax https://pennylane.ai/qml/demos/tutorial_jax_transformations.html

###################################################
###################################################
###################################################
args_symmetric = {'c': 2, 'e': 2, 'o': 1, 'm': 2, 'i': 1, 'd': 1}
args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4}

def data_encoding(game):
    '''
    loops through game array, applies RX(game[i]) on wire i
    input: 9x9 np array representation of ttt game
    output: None?

    0   1   2           0   1   2
    3   4   5    -->    7   8   3
    6   7   8           6   5   4
    '''
    fgame = game.flatten()
    order = [0, 1, 2, 5, 8, 7, 6, 3, 4]

    for i, j in enumerate(order):
        qml.RX(fgame[j], wires=i)


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

    
     # TODO: add ry
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
    connections = list(range(8)) + [0] 

    if symm:
        for i in range(8):
            qml.CRZ(param, wires=[connections[i], connections[i+1]])

    else:
        for i in range(8):
            qml.CRZ(param[i], wires=[connections[i], connections[i+1]])

def inner_layer(param, symm=True):
    '''
    entangles center qubit with edge qubits
    0    1    2
         |
    7  - 8 -  3
         |
    6    5    4
    '''
    connections = [1, 3, 5, 7]

    if symm:
        for i in connections:
            qml.CRZ(param, wires=[4, i])

    else:
        for n, i in enumerate(connections):
            qml.CRZ(param[n], wires=[4, i])    
  
def diag_layer(param, symm=True):
    '''
    entangles center qubit with corner qubits
    0    1    2 
      \     /
    7    8    3
      /     \ 
    6    5    4
    '''
    connections = [0, 2, 4, 6]

    if symm:
        for i in connections:
            qml.CRZ(param, wires=[8, i])

    else:
        for n, i in enumerate(connections):
            qml.CRZ(param[n], wires=[8, i])       

### Non symmetric functions ###
def data_encoding_old(game):
    '''
    loops through game array, applies RX(game[i]) on wire i
    input: 9x9 np array representation of ttt game
    output: None?
    '''
    fgame = game.flatten()
    for j in range(len(fgame)):
        qml.RX(fgame[j], wires=j)


def row_layer(params):
    '''
    entangles nearest neighbours qubits on each ROW of the game through CZs depending on params
    input: 6-elements parameters vector
    '''
    for row in range(3):
        for col in range(2):
            qml.CRZ(params[2*row+col],wires=[3*row+col,3*row+col+1])
        

def column_layer(params):
    '''
    entangles nearest neighbours qubits on each COLUMN of the game through CZs depending on params
    input: 6-elements parameters vector
    '''
    for col in range(3):
        for row in range(2):
            qml.CRZ(params[2*row+col],wires=[3*col+row,3*col+row+1])


def translate_to_parameters(design, symmetric=True):
    '''
    Translates circuit design string to number of required parameters in the form of a list corresponding to each applied layer
    '''
    param_args = [0]
    if symmetric:

        for i in design.replace(" ", ""):
            if i == 't':
                # encode data
                param_args.append(param_args[-1])
            elif i == 'c':
                # corners
                param_args.append(param_args[-1]+args_symmetric['c'])
            elif i == 'e':
                # edges
                param_args.append(param_args[-1]+args_symmetric['e'])
            elif i == 'o':
                # outer layer
                param_args.append(param_args[-1]+args_symmetric['o'])
            elif i == 'm':
                # center
                param_args.append(param_args[-1]+args_symmetric['m'])
            elif i == 'i':
                # inner layer
                param_args.append(param_args[-1]+args_symmetric['i'])
            elif i == 'd':
                # diagonal layer
                param_args.append(param_args[-1]+args_symmetric['d'])
    else:
        
        for i in design.replace(" ", ""):
            if i == 't':
                # encode data
                param_args.append(param_args[-1])
            elif i == 'c':
                # corners
                param_args.append(param_args[-1]+args_asymmetric['c'])
            elif i == 'e':
                # edges
                param_args.append(param_args[-1]+args_asymmetric['e'])
            elif i == 'o':
                # outer layer
                param_args.append(param_args[-1]+args_asymmetric['o'])
            elif i == 'm':
                # center
                param_args.append(param_args[-1]+args_asymmetric['m'])
            elif i == 'i':
                # inner layer
                param_args.append(param_args[-1]+args_asymmetric['i'])
            elif i == 'd':
                # diagonal layer
                param_args.append(param_args[-1]+args_asymmetric['d'])
        

    return param_args

#@qml.qnode(ttt_dev, interface='torch')
def circuit(game, params, symmetric, design="tceocem tceicem tcedcem"):
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

        args = translate_to_parameters(design)
        ngame = np.pi*0.5*game # normalize entries of game so they are between -pi/2, pi/2

        for r in range(params.shape[0]): # r repetitions
       

            for n, i in enumerate(design.replace(" ", "")):
                if i == 't':
                    data_encoding(ngame)
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


        return qml.expval(qml.PauliZ(8)) # measure one qubit in comp basis


full_circ = qml.QNode(circuit, ttt_dev)
full_circ_torch = qml.QNode(circuit, ttt_dev, interface='torch')
#full_circ_jax = qml.QNode(circuit, ttt_dev, interface='jax')
#full_circ = jax.jit(full_circ_jax)

###################################################
###################################################
###################################################

rng = np.random.default_rng(2021)

#TODO train something on a few labelled data!

############################

def random_params(repetitions, symmetric, design, enable_torch=False):
    '''
    returns array/torch tensor of paramters for amount of repetitions and circuit design
    '''
    params = rng.uniform(low=-1, high=1, size=(repetitions,translate_to_parameters(design, symmetric)[-1]))
    if enable_torch:
        return torch.tensor(params, requires_grad = True)
    else: 
        return np.array(params, requires_grad = True)


def cost_function(params,game, symmetric, design):
    return (full_circ(game,params, symmetric, design=design)-get_label(game))**2

def cost_function_batch(params,games_batch, symmetric, design):
    '''
    normalized least squares cost function over batch of data points (games)
    '''
    return sum([(full_circ(g,params, symmetric, design=design)-get_label(g))**2 for g in games_batch])/np.shape(games_batch)[0]

def cost_function_torch(params,game, symmetric, design):
    return (full_circ_torch(game,params, symmetric, design=design)-get_label(game))**2

def cost_function_batch_torch(params,games_batch, symmetric, design):
    '''
    normalized least squares cost function over batch of data points (games)
    '''
    return sum([(full_circ_torch(g,params, symmetric, design=design)-get_label(g))**2 for g in games_batch])/np.shape(games_batch)[0]

def gen_games_sample(size, wins=[1, 0, -1], output = None):
    '''
    Generates Tensor with 3*size games that are won equally by X, O and 0
    If the parameter output is a string instead of "None", the sample is stored in a npz file named after the string
    '''
    games_data,labels = get_data()

    sample = []
    sample_label = []
    
    for j in wins:
        sample += random.sample([a for k, a in enumerate(games_data) if labels[k] == j], size)
        sample_label += size*[j]
     
    if not output == None:
        with open(output+'.npz', 'wb') as f:
                np.savez(f,sample=sample, sample_label = sample_label)


    return np.tensor(sample, requires_grad=False), np.tensor(sample_label, requires_grad=False)

class tictactoe():

    def __init__(self, symmetric=True, sample_size=5, design="tceocem tceicem tcedcem", data_file=None):
        #self.opt = qml.GradientDescentOptimizer(0.01)
        self.sample_size = sample_size
        self.design = design

        if data_file == None:
            self.sample_games(sample_size)
        else:
            self.load_games(data_file, sample_size) # loads games and labels from file

        self.symmetric = symmetric

    def random_parameters(self, size=1, repetitions=2):
        '''
        sets random parameters for circuit design and amount of repetitions
        '''
        if size==1:
            self.init_params_torch = random_params(repetitions, self.symmetric, self.design, True)
            self.init_params = random_params(repetitions, self.symmetric, self.design)
        else:
            # Find best starting paramters
            params_list = [random_params(repetitions, self.symmetric, self.design, True) for i in range(size)]
            params_list_torch = [random_params(repetitions, self.symmetric, self.design) for i in range(size)]

            cost_list = [cost_function_batch(k,self.games_sample, self.symmetric) for k in params_list]
            self.init_params = params_list[np.argmin(cost_list)]
            self.init_params_torch = params_list_torch[np.argmin(cost_list)]

    def sample_games(self, size):
        '''
        Create random samples with equal amount of wins for X, O and 0
        '''
        self.games_sample, self.label_sample = gen_games_sample(size, wins=[-1, 0, 1])

    def load_games(self, data_file, size):
        '''
        Loads games and label from file specified by data_file. The first size data points are retained.
        If the file is not found, generate a new sample.
        Currently, load is implemented via numpy.load(file.npz)
        '''
        try:
            with open(data_file+'.npz', 'rb') as f:
                            print('Loading data file \n')
                            self.games_sample = np.load(f, allow_pickle = True)['sample'][:size]
                            self.label_sample = np.load(f, allow_pickle = True)['sample_label'][:size]
        except IOError: 
            print('Data sample not found, creating new one')
            self.sample_games(size)


    def run(self, steps, stepsize=0.01, resume = False):
        '''
        runs qml with standard pennylane gradient descent optimizer
        '''
        self.interface = 'pennylane'
        self.opt = qml.GradientDescentOptimizer(stepsize)

        if not resume:
            self.gd_cost = []
            self.theta = self.init_params
        for j in range(steps):
            self.theta = self.opt.step(lambda x: cost_function_batch(x, self.games_sample, self.symmetric, design=self.design),self.theta)
            cost_temp = cost_function_batch(self.theta,self.games_sample, self.symmetric, design=self.design)
            print(f"step {j} current cost value: {cost_temp}")
            self.gd_cost.append(cost_temp)   
            self.steps = j     

        print(self.gd_cost)
        print(self.theta)

    def run_lbgfs(self, steps, stepsize=0.1, resume = False):
        '''
        Runs qml with torch's lbfgs implementation. Usually converges much quicker than pennylanes standard gradient descent optimizer
        '''
        self.interface = 'torch'
        if not resume:
            self.gd_cost = []
            self.theta = self.init_params_torch

        self.opt = torch.optim.LBFGS([self.theta], lr=stepsize)

        def closure():
            self.opt.zero_grad()
            loss = cost_function_batch_torch(self.theta, self.games_sample, self.symmetric, self.design)
            loss.backward()
            return loss
        
        for j in range(steps):
            cost_temp = cost_function_batch_torch(self.opt.param_groups[0]['params'][0],self.games_sample, self.symmetric, self.design)
            print(f"step {j} current cost value: {cost_temp}")
            
            self.opt.step(closure)
            #print('step {}'.format(j))
            #print(self.opt.param_groups[0]['params'])

            self.gd_cost.append(cost_temp) 
            self.steps = j
        
        cost_temp = cost_function_batch_torch(self.opt.param_groups[0]['params'][0],self.games_sample, self.symmetric, self.design)
        print(f"final step current cost value: {cost_temp}")

        self.theta = self.opt.param_groups[0]['params'][0]  
            
    def check_accuracy(self, check_size = 100):
        '''
        checks accuracy of current theta by sampling check_size amount of games for each win
        '''
        games_check, labels_check = gen_games_sample(check_size)
        results = {-1: {}, 0: {}, 1: {}}
        results_alt = {-1: [], 0: [], 1: []}
        for i, game in enumerate(games_check[:500]):

            if self.interface == 'torch':
                res_device = round(float(full_circ_torch(game, self.theta, self.symmetric, design=self.design)), 3)
            else:
                res_device = round(float(full_circ(game, self.theta, self.symmetric, design=self.design)), 3)

            res_true = int(labels_check[i])
            results_alt[res_true].append(res_device)
            if res_device in results[res_true]:
                results[res_true][res_device] += 1
            else:
                results[res_true][res_device] = 1

        self.results = results

        # check accuracy
        self.accuracy = {-1: {}, 0: {}, 1: {}}

        self.accuracy[-1] = len([j for j in results_alt[-1] if (j <= -(1/3))])/len(results_alt[-1])
        self.accuracy[0] = len([j for j in results_alt[0] if ((j > -(1/3)) and (j < 1/3))])/len(results_alt[0])
        self.accuracy[1] = len([j for j in results_alt[1] if (j >= (1/3))])/len(results_alt[1])
        print('Accuracy for random sample of {} games: \n\n \t\t -1: {}% \n \t\t  0: {}% \n \t\t  1: {}%'.format(check_size*3, self.accuracy[-1]*100, self.accuracy[0]*100, self.accuracy[1]*100))

    def plot_cost(self):
        '''
        plots cost function
        '''
        plt.plot(self.gd_cost)
        plt.show()

    def save(self, name):
        '''
        saves result of qml as a npy file. Can be analyzed later
        '''
        to_save = {'symmetric': self.symmetric, 'accuracy': self.accuracy, 'steps': self.steps, 'design': self.design, 'interface': self.interface, 'cost function': self.gd_cost, 'sample size': self.sample_size, \
        'initial parameters': self.init_params.detach().numpy(), 'sampled games': self.games_sample.numpy(), 'theta': self.theta.detach().numpy()}
        #dd.io.save(name + '.h5', to_save)
        np.save(name, to_save)

#######################################################################################
#NOTE FROM FRANCESCO: I commented this to replace the following actions with run_ttt.py
#######################################################################################
'''
symmetric_run = tictactoeML()
asymetric_run = deepcopy(symmetric_run)
asymetric_run.symmetric = False

symmetric_run.random_parameters(20)
asymetric_run.random_parameters(20)

symmetric_run.run(100)
asymetric_run.run(100)

symmetric_run.check_accuracy()
asymetric_run.check_accuracy()

plt.plot(symmetric_run.gd_cost, label='symmetric')
plt.plot(asymetric_run.gd_cost, label='asymmetric')
plt.legend()
plt.show
'''

# TODO: run on cloud/cluster
# TODO: try different circuits

# TODO: enforce symmetries?
# %%
