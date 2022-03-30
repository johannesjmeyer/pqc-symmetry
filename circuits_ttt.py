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
#import deepdish as dd
#from jax.config import config
#config.update("jax_enable_x64", True)
#import jax
#import jax.numpy as jnp

###################################################
###################################################
###################################################
args_symmetric = {'c': 2, 'e': 2, 'o': 1, 'm': 2, 'i': 1, 'd': 1}
args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4}

test_var = 0
#if __name__ == '__main__':
#mp.set_start_method("spawn")


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
            qml.CRZ(param[0], wires=[connections[i], connections[i+1]])

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
            qml.CRZ(param[0], wires=[4, i])

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
            qml.CRZ(param[0], wires=[8, i])

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
def circuit(game, params, symmetric, design="tceocem tceicem tcedcem", alt_results=True):
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

        if alt_results:
            return [qml.expval(qml.PauliZ(i)) for i in range(9)]
        else:
            return qml.expval(qml.PauliZ(8)) # measure one qubit in comp basis

ttt_dev = qml.device("default.qubit", wires=9) # the device used to label ttt instances 
#ttt_dev = qml.device("lightning.qubit", wires=9) # ligthing.qubit is an optimized version of default.qubit
# TODO: use other device https://pennylane.ai/plugins.html

full_circ = qml.QNode(circuit, ttt_dev)
#full_circ_torch = qml.QNode(circuit, ttt_dev, interface='torch', diff_method="adjoint")# is supoosed to limit RAM usage
full_circ_torch = qml.QNode(circuit, ttt_dev, interface='torch')

###################################################
###################################################
###################################################


rng = np.random.default_rng(2021)

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

def cost_function(circ, params,game, symmetric, design):
    return (circ(game,params, symmetric, design=design)-get_label(game))**2

def cost_function_batch(circ, params,games_batch, symmetric, design):
    '''
    normalized least squares cost function over batch of data points (games)
    '''
    return sum([(circ(g,params, symmetric, design=design)-get_label(g))**2 for g in games_batch])/np.shape(games_batch)[0]

def cost_function_alt(circ, params, game, symmetric, design="tceocem tceicem tcedcem"):

    result = circ(game,params, symmetric, design=design, alt_results=True)
    print(result)
    label = get_label(game)
    slicer = [[0, 2, 4, 6], [8], [1, 3, 5, 7]]
    
    won = np.zeros(3)
    won[label+1] = 1
    print(won)

    avg_result = []
    for i in np.array([-1, 0, 1])+1:
        print((np.average(result[slicer[i]]) - won[i])**2)
        avg_result.append((np.average(result[slicer[i]]) - won[i])**2)
    
    return np.sum(avg_result)

def cost_function_alt_batch(circ, params, games, symmetric, design="tceocem tceicem tcedcem"):
    
    final_results = torch.zeros(len(games))
    for i, g in enumerate(games):  # TODO: implement multiprocessing with pool here

        result = circ(g,params, symmetric, design=design, alt_results=True)
        label = get_label(g)
        
        won = torch.zeros(3)
        won[label+1] = 1

        avg_result = get_results(result)

        final_results[i] = torch.sum((avg_result - won)**2)

    return torch.mean(final_results)

def cross_entropy_cost_batch(circ, params, games, symmetric, design):

    loss = torch.nn.CrossEntropyLoss()
    input = torch.zeros((len(games), 3))
    target = torch.zeros(len(games), dtype=torch.long)
    for i, g in enumerate(games):
        result = circ(g,params, symmetric, design=design, alt_results=True)
        input[i] = get_results_no_normalizing(result)
        target[i] = int(get_label(g) + 1)

    return loss(input, target)

    """"
    cost_vector = torch.zeros(len(games))
    for i, g in enumerate(games):

        result = circ(g,params, symmetric, design=design, alt_results=True)
        label = get_label(g)
        
        y = torch.zeros(3)
        y[label+1] = 1
        a = get_results(result)

        #a = result
        #slicer = [[0, 2, 4, 6], [8], [1, 3, 5, 7]]
        #y = torch.zeros(9)
        #y[slicer[label+1]] = 1

        cost_vector[i] = torch.sum(y*torch.log(a)+(1-y)*torch.log(1-a))

    return -torch.mean(cost_vector)
    """

def get_results_no_normalizing(result):
    avg_result = []
    slicer = [[0, 2, 4, 6], [8], [1, 3, 5, 7]]
    for i in np.array([-1, 0, 1])+1:
        avg_result.append(torch.mean(result[slicer[i]]).reshape(1))

    result = torch.cat(avg_result)
    #logistic layer
    #result = torch.exp(a) 
    #result = result/torch.linalg.norm(result)

    return result

def get_results(result):
    avg_result = []
    slicer = [[0, 2, 4, 6], [8], [1, 3, 5, 7]]
    for i in np.array([-1, 0, 1])+1:
        avg_result.append(torch.mean(result[slicer[i]]).reshape(1))

    result = (torch.cat(avg_result) + 1)/2
    #logistic layer
    #result = torch.exp(a) 
    #result = result/torch.linalg.norm(result)

    return result

games_data,labels = get_data()
games_data_reduced, labels_reduced = get_data_symmetric()

def gen_games_sample(size, wins=[1, 0, -1], output = None, reduced=False, truesize=False):
    '''
    Generates Tensor with 3*size games that are won equally by X, O and 0
    If the parameter output is a string instead of "None", the sample is stored in a npz file named after the string

    param wins: list of wins to be included in sample. If empty, returns completely random sample.
    '''
    sample = []
    sample_label = []
    #print('Generating new samples...')
    if reduced:
        data = games_data_reduced
        data_labels = labels_reduced
    else: 
        data = games_data
        data_labels = labels
    if wins:
        if truesize:
            for i in range(size):
                sample += [random.choice([a for k, a in enumerate(data) if data_labels[k] == i%len(wins)-1])]
                sample_label += [i%len(wins)-1]
        else:
            for j in wins:
                sample += random.sample([a for k, a in enumerate(data) if data_labels[k] == j], size)
                sample_label += size*[j]
    else:
        sample += random.sample(list(data), size)
        sample_label = [get_label(g) for g in sample]
        #sample_label += size*[j]
        
    if not output == None:
        with open('samples/'+output+'.npz', 'wb') as f:
                np.savez(f,sample=sample)#, sample_label = sample_label)


    return np.tensor(sample, requires_grad=False), np.tensor(sample_label, requires_grad=False)

class tictactoe():

    def __init__(self, symmetric=True, sample_size=5, design="tceocem tceicem tcedcem", data_file=None, alt_results=True, random_sample=False, random_wins = False, reduced = False, cross_entropy = False):

        self.sample_size = sample_size
        self.design = design
        self.alt_results = alt_results
        self.random = random_sample
        self.random_wins = random_wins
        self.reduced = reduced
        self.epochs = False

        if cross_entropy:
            print('using cross entropy cost function...')
            self.cost_function = cross_entropy_cost_batch
        elif alt_results:
            print('using mean squared cost function...')
            self.cost_function = cost_function_alt_batch
        else:
            self.cost_function = cost_function_batch

        if data_file is None:
            self.sample_games(sample_size)
        else:
            self.load_games(data_file, sample_size) # loads games and labels from file

        self.symmetric = symmetric

    def random_parameters(self, size=1, repetitions=2):
        '''
        sets random parameters for circuit design and amount of repetitions
        '''
        self.repetitions = repetitions
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
        if self.random_wins:
            wins = []
        else:
            wins = [-1, 0, 1]
        self.games_sample , self.label_sample = gen_games_sample(size, wins=wins, reduced = self.reduced)

    def load_games(self, data_file, size):
        '''
        Loads games and label from file specified by data_file. The first size data points are retained.
        If the file is not found, generate a new sample.
        Currently, load is implemented via numpy.load(file.npz)
        '''
        try:
            with open('samples/'+data_file+'.npz', 'rb') as f:
                            print('Loading data file \n')
                            self.games_sample = np.tensor(np.load(f, allow_pickle = True)['sample'][:size*3], requires_grad=False)
                            #self.label_sample = np.tensor(np.load(f, allow_pickle = True)['sample_label'][:size*3], requires_grad=False)
        except IOError: 
            print('Data sample not found, creating new one')
            self.sample_games(size)


    def run_epochs(self, epochs, samplesize_per_step, steps_per_epoch, stepsize, multiplicator=1, data_file = None):
        """
        Runs lbfgs training with different epochs
        """
        self.epochs = True
        if data_file is None:
            if multiplicator == 1:
                self.batch = gen_games_sample(steps_per_epoch*samplesize_per_step, truesize=True, reduced = self.reduced)[0]
            else:
                games_per_batch = int(np.ceil(steps_per_epoch*samplesize_per_step / multiplicator))
                batch = gen_games_sample(games_per_batch, truesize=True, reduced = self.reduced)[0]
                self.batch = batch
                for i in range(int(np.ceil(multiplicator))):
                    self.batch = np.append(self.batch, batch, axis=0)
                np.random.shuffle(self.batch)
                self.batch = self.batch[:steps_per_epoch*samplesize_per_step]
        else:
            with open('samples/'+data_file+'.npz', 'rb') as f:
                            print('Loading data file \n')
                            self.batch = np.tensor(np.load(f, allow_pickle = True)['sample'], requires_grad=False)

        self.interface = 'torch'

        self.gd_cost = []
        self.theta = self.init_params_torch

        self.opt = torch.optim.LBFGS([self.theta], lr=stepsize)
        self.stepsize = stepsize
        self.epoch_total_accuracy = []
        self.epoch_accuracy = []
        self.epoch_cost_function = []

        def closure():
            self.opt.zero_grad()
            loss = self.cost_function(full_circ_torch, self.theta, self.games_sample, self.symmetric, self.design)
            loss.backward()
            return loss
        
        step_start = 0
        step_end = 0
        
        for i in range(epochs):

            for j, sample in enumerate([self.batch[k:k+samplesize_per_step] for k in range(0, len(self.batch), samplesize_per_step)]):
                self.games_sample = sample
                cost_temp = self.cost_function(full_circ_torch, self.opt.param_groups[0]['params'][0], self.games_sample, self.symmetric, self.design)
                print(f"epoch {i}/{epochs} step {j}/{steps_per_epoch} current cost value: {cost_temp} execution time: {step_end-step_start}s")
                step_start = timer()
                self.opt.step(closure)
                step_end = timer()

                self.gd_cost.append(cost_temp) 
                self.steps = j
            
            np.random.shuffle(self.batch)
            print(f'unique games: {len(np.unique(self.batch, axis=0))}')
            self.theta = self.opt.param_groups[0]['params'][0]  
            print(f'epoch {i}/{epochs} accuracy:')
            self.epoch_accuracy.append(self.check_accuracy(check_batch=self.batch))
            print(f'epoch {i}/{epochs} total accuracy:')
            self.epoch_total_accuracy.append(self.check_accuracy())
            self.epoch_cost_function.append(self.cost_function(full_circ_torch, self.opt.param_groups[0]['params'][0], self.batch, self.symmetric, self.design))
            #print(f"epoch {i}/{epochs} accuracy: {self.epoch_accuracy[-1]}")
            print(f"epoch {i}/{epochs} cost function: {self.epoch_cost_function[-1]}")
        
        print('Done')
        self.theta = self.opt.param_groups[0]['params'][0]  


    def run(self, steps, stepsize=0.01, resume = False):
        '''
        runs qml with standard pennylane gradient descent optimizer
        '''
        if self.alt_results:
            raise NotImplementedError('Alt Results only available for torch lbfgs as of now')

        self.interface = 'pennylane'
        self.opt = qml.GradientDescentOptimizer(stepsize)
        self.stepsize = stepsize

        if not resume:
            self.gd_cost = []
            self.theta = self.init_params
        for j in range(steps):
            self.theta = self.opt.step(lambda x: self.cost_function(full_circ, x, self.games_sample, self.symmetric, design=self.design),self.theta)
            cost_temp = self.cost_function(full_circ, self.theta,self.games_sample, self.symmetric, design=self.design)

            if self.random:
                self.sample_games(self.sample_size)
            print(f"step {j} current cost value: {cost_temp}")
            self.gd_cost.append(cost_temp)   
            self.steps = j     
        #print(self.gd_cost)
        #print(self.theta)

    def run_lbgfs(self, steps, stepsize=0.1, resume = False):
        '''
        Runs qml with torch's lbfgs implementation. Usually converges much quicker than pennylanes standard gradient descent optimizer
        '''
        print('running lbgfs...')
        print(tabulate([['steps', steps], ['stepsize', stepsize], ['symmetric', self.symmetric], ['design', self.design], ['sample size', 3*self.sample_size], \
            ['random sample', self.random], ['repetitions', self.repetitions]]))

        self.interface = 'torch'
        if not resume:
            self.gd_cost = []
            self.theta = self.init_params_torch

        self.opt = torch.optim.LBFGS([self.theta], lr=stepsize)
        self.stepsize = stepsize

        def closure():
            self.opt.zero_grad()
            loss = self.cost_function(full_circ_torch, self.theta, self.games_sample, self.symmetric, self.design)
            loss.backward()
            return loss

        step_start = 0
        step_end = 0
        
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
            self.steps = j
        
        cost_temp = self.cost_function(full_circ_torch ,self.opt.param_groups[0]['params'][0],self.games_sample, self.symmetric, self.design)
        print(f"final step current cost value: {cost_temp}")

        self.theta = self.opt.param_groups[0]['params'][0]  
            
    def check_accuracy(self, check_size = 100, check_batch = None): # TODO: check if accuracy varies for same run
        # TODO: implement confusion matrix
        '''
        checks accuracy of current theta by sampling check_size amount of games for each win
        '''
        if check_batch is None:
            games_check, labels_check = gen_games_sample(check_size)
        else: 
            games_check = check_batch
            labels_check = np.tensor([get_label(i) for i in check_batch])

        results = {-1: {}, 0: {}, 1: {}}
        results_alt = {-1: [], 0: [], 1: []}
        res_circ = []
        res_true = []

        for i, game in enumerate(games_check):

            res_true.append(int(labels_check[i]))

            if self.alt_results:
                res = full_circ_torch(game, self.theta, self.symmetric, design=self.design, alt_results=True)

                avg_results = get_results(res)

                res_circ.append(avg_results.detach().numpy())

            else:
                if self.interface == 'torch':
                    res_device = round(float(full_circ_torch(game, self.theta, self.symmetric, design=self.design)), 3)
                else:
                    res_device = round(float(full_circ(game, self.theta, self.symmetric, design=self.design)), 3)

                results_alt[res_true[i]].append(res_device)
                if res_device in results[res_true[i]]:
                    results[res_true[i]][res_device] += 1
                else:
                    results[res_true[i]][res_device] = 1

        #self.results = results

        # check accuracy

        if self.alt_results:
            # confusion matrix:
            won = [-1, 0, 1]
            res_circ2 = [won[i.argmax()] for i in res_circ]
            self.confusion_matrix = confusion_matrix(res_true, res_circ2, normalize='true')
            self.accuracy = self.confusion_matrix.trace()/3
            print('Confusion matrix:')
            print(self.confusion_matrix)

            return self.confusion_matrix
            
        else:
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

    def save(self, name, exec_time=0):
        '''
        saves result of qml as a npy file. Can be analyzed later
        '''
        if self.interface == 'torch':
            params_tmp = self.init_params_torch.detach().numpy()
            theta_tmp = self.theta.detach().numpy()
        elif self.interface == 'pennylane':
            params_tmp = self.init_params.numpy()
            theta_tmp = self.theta.numpy()
        to_save = {'symmetric': self.symmetric, 'alt result': self.alt_results,'accuracy': self.confusion_matrix,'execution time': exec_time, 'steps': self.steps, 'stepsize': self.stepsize, 'design': self.design, 'interface': self.interface, 'cost function': self.gd_cost, 'sample size': self.sample_size, \
        'initial parameters': params_tmp, 'sampled games': self.games_sample.numpy(), 'theta': theta_tmp}
        if self.epochs:
            to_save['epoch cost'] = self.epoch_cost_function
            to_save['epoch accuracy'] = self.epoch_accuracy
            to_save['epoch batch'] = self.batch
            to_save['epoch total accuracy'] = self.epoch_total_accuracy

        #dd.io.save(name + '.h5', to_save)
        print('Saving results as {}.npy'.format(name))
        try:
            np.save('output/'+name, to_save)
        except FileNotFoundError:
            os.makedirs(os.getcwd()+'/output/'+name[::-1].split('/', 1)[1][::-1])
            np.save('output/'+name, to_save)
# TODO: run on cloud/cluster
# TODO: try different circuits

# %%
# %%
