# %%
from cmath import pi
from http.client import responses
import pennylane as qml
from pennylane import numpy as np
from tictactoe import *
from tabulate import tabulate
from timeit import default_timer as timer 
import os
from sklearn.metrics import confusion_matrix
import torch
#import deepdish as dd

###################################################
###################################################
###################################################

if __name__ == "__main__":
    args_symmetric = {'c': 2, 'e': 2, 'o': 1, 'm': 2, 'i': 1, 'd': 1}
    args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4}
    gate_2q = qml.CRX
    rotation_2q = True
    corner_qubits = [0, 2, 4, 6]
    edge_qubits = [1, 3, 5, 7]
    middle_qubit = [8]

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
    if symm:
        for i in corner_qubits:

            qml.RX(param[0], wires=i)
            qml.RY(param[1], wires=i)

    else:
        for n, i in enumerate(corner_qubits):
            qml.RX(param[i], wires=i)
            qml.RY(param[i+1], wires=i)

    
     # TODO: add ry
def edges(param, symm=True):
    '''
    applies RX an RZ gates on edges
    '''
    if symm:
        for i in edge_qubits:

            qml.RX(param[0], wires=i)
            qml.RY(param[1], wires=i)

    else:
        for n, i in enumerate(edge_qubits):
            qml.RX(param[i-1], wires=i)
            qml.RY(param[i], wires=i)
    
    
def center(param):
    '''
    applies RX an RZ gate on the middle qubit
    '''
    qml.RX(param[0], wires=middle_qubit[0])
    qml.RY(param[1], wires=middle_qubit[0])


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
    if rotation_2q:
        if symm:
            for i in range(4):
                    gate_2q(param[0], wires=[corner_qubits[i], edge_qubits[i]])
                    gate_2q(param[0], wires=[corner_qubits[i], edge_qubits[i-1]])
        else:
            for i in range(4):
                    gate_2q(param[2*i], wires=[corner_qubits[i], edge_qubits[i]])
                    gate_2q(param[2*i+1], wires=[corner_qubits[i], edge_qubits[i-1]])
    else:
        for i in range(4):
            gate_2q(wires=[corner_qubits[i], edge_qubits[i]])
            gate_2q(wires=[corner_qubits[i], edge_qubits[i-1]])      


def inner_layer(param, symm=True):
    '''
    entangles center qubit with edge qubits
    0    1    2
         |
    7  - 8 -  3
         |
    6    5    4
    '''
    if rotation_2q:
        if symm:
            for i in edge_qubits:
                gate_2q(param[0], wires=[i, middle_qubit[0]])
        else:
            for n, i in enumerate(edge_qubits):
                gate_2q(param[n], wires=[i, middle_qubit[0]])
    else:
        for i in edge_qubits:
            gate_2q(wires=[i, middle_qubit[0]]) 

  
def diag_layer(param, symm=True):
    '''
    entangles center qubit with corner qubits
    0    1    2 
      \     /
    7    8    3
      /     \ 
    6    5    4
    '''
    if rotation_2q:
        if symm:
            for i in corner_qubits:
                gate_2q(param[0], wires=[middle_qubit[0], i])
        else:
            for n, i in enumerate(corner_qubits):
                gate_2q(param[n], wires=[middle_qubit[0], i])
    else:
        for i in corner_qubits:
            gate_2q(wires=[middle_qubit[0], i])

def specify_gates(controlstring):
    '''
    replaces the control string fed in input with the appropriate pennylane gates used to build the circuit
    '''
    global rotation_2q
    global gate_2q
    global args_symmetric
    global args_asymmetric

    if controlstring == 'rx':
        rotation_2q = True
        gate_2q = qml.CRX

    elif controlstring == 'rz':
        rotation_2q = True
        gate_2q = qml.CRZ

    elif controlstring == 'ry':
        rotation_2q = True
        gate_2q = qml.CRY

    elif controlstring == 'x':
        rotation_2q = False
        gate_2q = qml.CNOT

    elif controlstring == 'z':
        rotation_2q = False
        gate_2q = qml.CZ

    else:
        raise TypeError

    if rotation_2q:
        args_symmetric = {'c': 2, 'e': 2, 'o': 1, 'm': 2, 'i': 1, 'd': 1}
        args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4}
    else:
        args_symmetric = {'c': 2, 'e': 2, 'o': 0, 'm': 2, 'i': 0, 'd': 0}
        args_asymmetric = {'c': 8, 'e': 8, 'o': 0, 'm': 2, 'i': 0, 'd': 0}
        



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

        args = translate_to_parameters(design, symmetric)
        ngame = np.pi*game*2/3  # normalize entries of game so they are between -2pi/3, 2pi/3

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


        return [qml.expval(qml.PauliZ(i)) for i in range(9)]


#ttt_dev = qml.device("default.qubit", wires=9) # the device used to label ttt instances 
ttt_dev = qml.device("lightning.qubit", wires=9) # ligthing.qubit is an optimized version of default.qubit


full_circ = qml.QNode(circuit, ttt_dev)
full_circ_torch = qml.QNode(circuit, ttt_dev, interface='torch', diff_method="adjoint")# is supoosed to limit RAM usage
#full_circ_torch = qml.QNode(circuit, ttt_dev, interface='torch')

###################################################
###################################################
###################################################


rng = np.random.default_rng()
# rng = np.random.default_rng(2021) # for reproducible results

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


def cost_function_alt_batch(circ, params, games, symmetric, design="tceocem tceicem tcedcem"):
    
    final_results = torch.zeros(len(games))
    for i, g in enumerate(games):  # TODO: implement multiprocessing with pool here

        result = circ(g,params, symmetric, design=design)
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
        result = circ(g,params, symmetric, design=design)
        input[i] = get_results_no_normalizing(result)
        target[i] = int(get_label(g) + 1)

    return loss(input, target)


def cost_fn_5q(circ, params, games, symmetric, design):

    loss = 0
    for i, g in enumerate(games):  # TODO: implement multiprocessing with pool here
        result = circ(g,params, symmetric, design=design)
        label = get_label(g)
        avg_result = get_results(result)

        if label == 0:
            loss += (avg_result[1]-1)**2
        elif label == 1:
            loss += ((avg_result[1]+1)**2 + (avg_result[0]-1)**2)/2
        elif label == -1:
            loss += ((avg_result[1]+1)**2 + (avg_result[0]+1)**2)/2
        else:
            raise TypeError('game has no label')      

    return loss/len(games)


def get_results_no_normalizing(result):
    '''
    takes 9 qubits exp values and averages edges (4q)/corners (4q)/center (1q), returns 3d vector
    only used for cross-entropy loss function
    '''
    # THIS SHOULD ALREADY BE GOOD FOR TORCH (CE)
    avg_result = []
    slicer = [corner_qubits, middle_qubit, edge_qubits]
    for i in np.array([-1, 0, 1])+1:
        avg_result.append(torch.mean(result[slicer[i]]).reshape(1))

    result = torch.cat(avg_result)

    return result

def get_results(result):
    '''same as above but normalized to [0,1]'''
    avg_result = []
    slicer = [corner_qubits, middle_qubit, edge_qubits]
    for i in np.array([-1, 0, 1])+1:
        avg_result.append(torch.mean(result[slicer[i]]).reshape(1))

    result = (torch.cat(avg_result) + 1)/2

    return result

games_data,labels = get_data()
games_data_reduced, labels_reduced = get_data_symmetric()

def gen_games_sample(size, wins=[1, 0, -1], output = None, reduced=False, truesize=False):
    '''
    Generates Tensor with 3*size games that are won equally by X, O and 0
    If the parameter output is a string instead of "None", the sample is stored in a npz file named after the string

    param wins: list of wins to be included in sample. If empty, returns completely random sample.

    NOTE: truesize is only used for lbfgs, otherwise the number of games in the set is fixed in the case of epochs
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
            partial_size = int(np.ceil(size/(len(wins))))
            for j in wins:
                sample += list(rng.choice(data[data_labels == j], partial_size,replace=False))
                sample_label += partial_size*[j]

            sample = sample[:size]
            sample_label = sample_label[:size]
        else:
            for j in wins:
                sample += list(rng.choice(data[data_labels == j], size, replace=False))
                sample_label += size*[j]
    else:
        sample += list(rng.choice(data, size,replace=False))
        sample_label = [get_label(g) for g in sample]
        
    if not output == None:
        with open('samples/'+output+'.npz', 'wb') as f:
                np.savez(f,sample=sample)#, sample_label = sample_label)


    return np.tensor(sample, requires_grad=False), np.tensor(sample_label, requires_grad=False)

class tictactoe():

    def __init__(self, symmetric=True, sample_size=5, design="tceocem tceicem tcedcem", data_file=None, random_sample=False, wins = [-1, 0, 1], reduced = False, cross_entropy = False, cost_5q=False):

        self.sample_size = sample_size
        self.design = design
        self.random = random_sample
        self.wins = wins
        self.reduced = reduced
        self.epochs = False # used for saving results, turns to True if run_epochs happens
        self.cost_5q = cost_5q

        if cross_entropy:
            print('using cross entropy cost function...')
            self.cost_function = cross_entropy_cost_batch
        elif cost_5q:
            print('using 5q cost function...')
            self.cost_function = cost_fn_5q
        else:
            print('using mean squared cost function...')
            self.cost_function = cost_function_alt_batch

        if data_file is None:
            self.sample_games(sample_size)
        else:
            self.load_games(data_file, sample_size) # loads games and labels from file

        self.symmetric = symmetric

    def random_parameters(self, repetitions=2):
        '''
        sets random parameters for circuit design and amount of repetitions
        '''
        self.repetitions = repetitions
        self.init_params_torch = random_params(repetitions, self.symmetric, self.design, True)
        self.init_params = random_params(repetitions, self.symmetric, self.design)

    def sample_games(self, size):
        '''
        Create random samples with equal amount of wins for X, O and 0
        '''
        self.games_sample , self.label_sample = gen_games_sample(size, wins=self.wins, reduced = self.reduced)

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


    def run_epochs(self, epochs, samplesize_per_step, steps_per_epoch, stepsize, data_file = None):
        """
        Runs Adam training with different epochs
        """
        print('running epochs...')
        print(tabulate([['epochs', epochs], ['stepsize', stepsize], ['symmetric', self.symmetric], ['design', self.design], ['sample size per step', samplesize_per_step], ['steps per epoch', steps_per_epoch], \
            ['wins', self.wins], ['repetitions', self.repetitions], ['corners', corner_qubits], ['edges', edge_qubits], ['center', middle_qubit]]))

        self.epochs = True
        if data_file is None:
            self.batch = gen_games_sample(size = steps_per_epoch*samplesize_per_step, truesize = True, reduced = self.reduced, wins=self.wins)[0]
            
        else:
            with open('samples/'+data_file+'.npz', 'rb') as f:
                            print('Loading data file \n')
                            self.batch = np.tensor(np.load(f, allow_pickle = True)['sample'], requires_grad=False)

        self.test_batch = gen_games_sample(size = 600, truesize = True, reduced = self.reduced, wins = self.wins)[0]

        np.random.shuffle(self.batch)

        self.interface = 'torch'

        self.gd_cost = []
        self.theta = self.init_params_torch
        self.opt = torch.optim.Adam([self.theta], lr=stepsize)
        self.stepsize = stepsize
        self.epoch_total_accuracy = []
        self.epoch_accuracy = []
        self.epoch_cost_function = []

        step_start = 0
        step_end = 0
        
        print(f'epoch 0/{epochs} accuracy:')
        self.epoch_accuracy.append(self.check_accuracy(check_batch=self.batch))
        print(f'epoch 0/{epochs} total accuracy:')
        self.epoch_total_accuracy.append(self.check_accuracy(check_batch=self.test_batch))
        self.epoch_cost_function.append(self.cost_function(full_circ_torch, self.theta, self.batch, self.symmetric, self.design))
          

        for i in range(1,epochs+1):

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
            print(f'unique games: {len(np.unique(self.batch, axis=0))}')
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
            ['random sample', self.random], ['repetitions', self.repetitions], ['wins', self.wins]]))

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
        print('accuracy on training data set:')
        self.check_accuracy(check_batch=self.games_sample)
        self.theta = self.opt.param_groups[0]['params'][0]  
            
    def check_accuracy(self, check_size = None, check_batch = None):
        '''
        checks accuracy of current theta by sampling check_size amount of games for each win
        NOTE: if check_batch is specified, check 
        '''

        if (check_size is None) and (check_batch is None):
            raise AttributeError('check_size and check_batch cannot both be None')

        if check_batch is None:
            games_check, labels_check = gen_games_sample(check_size, wins=self.wins)
        else: 
            games_check = check_batch
            labels_check = np.tensor([get_label(i) for i in check_batch])

        res_circ = []
        res_true = []

        for i, game in enumerate(games_check):

            res_true.append(int(labels_check[i]))

            res = full_circ_torch(game, self.theta, self.symmetric, design=self.design)

            avg_results = get_results(res)

            res_circ.append(avg_results.detach().numpy())

        # check accuracy
        if self.cost_5q:
            won = [-1, 0, 1]
            res_circ2 = []
            for i in res_circ:
                if i[1] < 0:
                    res_circ2.append(0)
                elif i[0] > 0:
                    res_circ2.append(1)
                else: 
                    res_circ.append(-1)

            self.confusion_matrix = confusion_matrix(res_true, res_circ2, normalize='true')
            self.accuracy = self.confusion_matrix.trace()/3
            print('Confusion matrix:')
            print(self.confusion_matrix)

            return self.confusion_matrix

        else:
            # confusion matrix:
            won = [-1, 0, 1]
            res_circ2 = [won[i.argmax()] for i in res_circ]
            self.confusion_matrix = confusion_matrix(res_true, res_circ2, normalize='true')
            self.accuracy = self.confusion_matrix.trace()/3
            print('Confusion matrix:')
            print(self.confusion_matrix)

            return self.confusion_matrix
        

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
        to_save = {'symmetric': self.symmetric, 'accuracy': self.confusion_matrix,'execution time': exec_time, 'steps': self.steps, 'stepsize': self.stepsize, 'design': self.design, 'interface': self.interface, 'cost function': self.gd_cost, 'sample size': self.sample_size, \
        'initial parameters': params_tmp, 'sampled games': self.games_sample.numpy(), 'theta': theta_tmp}
        if self.epochs:
            to_save['epoch cost'] = self.epoch_cost_function
            to_save['epoch accuracy'] = self.epoch_accuracy
            to_save['epoch batch'] = self.batch
            to_save['epoch total accuracy'] = self.epoch_total_accuracy

        #dd.io.save(name + '.h5', to_save)
        print('Saving results as {}.npy'.format(name))
        try:
            #np.save('output/'+name, to_save)
            np.save(name, to_save)
        except FileNotFoundError:
            os.makedirs(os.getcwd()+'/output/'+name[::-1].split('/', 1)[1][::-1])
            np.save('output/'+name, to_save)

# %%
# %%
