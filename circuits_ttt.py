# %%
from cmath import pi
from http.client import responses
from tkinter.tix import Y_REGION
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
    '''
    Only relevant if run as main file. Normally these variables are initialized by run_ttt.py
    '''
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

    default:
        0    -    2
                
        -    -    -
                
        6    -    4
    '''
    if symm:
        for i in corner_qubits:
            qml.RX(param[0], wires=i)
            qml.RY(param[1], wires=i)

    else:
        for n, i in enumerate(corner_qubits):
            qml.RX(param[2*n], wires=i)
            qml.RY(param[2*n+1], wires=i)

    
     # TODO: add ry
def edges(param, symm=True):
    '''
    applies RX an RZ gates on edges

    default:
        -    1    -
               
        7    -    3
                
        -    5    -
    '''
    if symm:
        for i in edge_qubits:
            qml.RX(param[0], wires=i)
            qml.RY(param[1], wires=i)

    else:
        for n, i in enumerate(edge_qubits):
            qml.RX(param[2*n], wires=i)
            qml.RY(param[2*n + 1], wires=i)
    
    
def center(param):
    '''
    applies RX an RZ gate on the middle qubit
    default:
        -    -    -
                
        -    8    -
                
        -    -    -
    '''
    qml.RX(param[0], wires=middle_qubit[0])
    qml.RY(param[1], wires=middle_qubit[0])


def outer_layer(param, symm=True):
    '''
    entangles nearest neighboar qubits of the outer layer of the tictactoe board

    default:
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

    default:
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

    default:
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
    translates design string to list of numbers of parameter for each sub layer for symmetric or asymnetric case
    '''
    param_args = [0]
    if symmetric:
        args = args_asymmetric
    else:
        args = args_asymmetric

    for i in design.replace(" ", ""):
        if i == 't':
            # encode data
            param_args.append(param_args[-1])
        else:
            # layer 'i'
            param_args.append(param_args[-1]+args[i])

    return param_args  

def circuit(game, params, symmetric, design="tcemoid"):
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

# TODO: lightning.qubit only seems to work with linux. Add warning?

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

def random_params(repetitions, symmetric, design):
    '''
    returns array/torch tensor of paramters for amount of repetitions and circuit design
    '''
    params = rng.uniform(low=-1, high=1, size=(repetitions,translate_to_parameters(design, symmetric)[-1]))

    return torch.tensor(params, requires_grad = True)


def loss_MSE(circ, params, games, symmetric, design="tceocem tceicem tcedcem"):
    '''
    computes the mean squared loss of given games and parameters
    '''
    L = torch.zeros(len(games))
    for i, g in enumerate(games):  # TODO: implement multiprocessing with pool here

        result = circ(g,params, symmetric, design=design)
        label = get_label(g)
        
        y = torch.zeros(3)
        y[label+1] = 1

        y_g = get_results(result)

        L[i] = torch.sum((y_g - y)**2)

    return torch.mean(L)

def loss_CE(circ, params, games, symmetric, design):
    '''
    computes the cross entropy loss of given games and parameters
    '''
    loss = torch.nn.CrossEntropyLoss()
    input = torch.zeros((len(games), 3))
    target = torch.zeros(len(games), dtype=torch.long)
    for i, g in enumerate(games):
        result = circ(g,params, symmetric, design=design)
        input[i] = get_results_no_normalizing(result)
        target[i] = int(get_label(g) + 1)

    return loss(input, target)


def loss_5q(circ, params, games, symmetric, design):
    '''
    computes custom mean squared loss of given games and parameters only on 5 qubits
    Tie/Win is encoded in middle qubit (Z_8 > 0 --> Tie, Z_8 < 0 --> Win)
    If Tie, all other results are ignored.
    If Win, win for X or O is encoded in corner qubits (Z_c > 0 --> X, Z_c < 0 --> O)
    '''
    loss = 0
    for i, g in enumerate(games):
        result = circ(g, params, symmetric, design=design)
        label = get_label(g)
        y_g = get_results_no_normalizing(result)

        if label == 0:
            loss += (y_g[1]-1)**2
        elif label == 1:
            loss += ((y_g[1]+1)**2 + (y_g[0]-1)**2)/2
        elif label == -1:
            loss += ((y_g[1]+1)**2 + (y_g[0]+1)**2)/2
        else:
            raise TypeError('game has no label')      

    return loss/len(games)


def get_results_no_normalizing(result):
    '''
    takes 9 qubits exp values and averages edges (4q)/corners (4q)/center (1q), returns 3d vector
    only used for cross-entropy and 5q loss function
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

def load_data(reduced):
    '''Loads all games and labels into global namespace
    reduced : bool
        if true, only loads games that are unique in symmetry group
    '''
    global games_data, labels
    if reduced:
        games_data, labels = get_data_symmetric()
    else:
        games_data,labels = get_data()

def gen_games_sample(size, wins=[1, 0, -1], output = None):
    '''Generates Tensor with size games that are won equally by X, O and 0
    If the parameter output is a string instead of "None", the sample is stored in a npz file named after the string

    size : int
        amout of games sampled
    wins : list
        list of wins to be included in sample. If empty, returns completely random sample.
    output : str (optional)
        if spceified, saves generated games as .npz file
    '''
    sample = []
    sample_label = []

    data = games_data
    data_labels = labels

    if wins:
        partial_size = int(np.ceil(size/(len(wins))))

        for j in wins:
            sample += list(rng.choice(data[data_labels == j], partial_size,replace=False))
            sample_label += partial_size*[j]

        if len(sample) != size:
            print('Warning: size is not a multiple of len(wins), uneven distribution of games!')

        sample = sample[:size]
        sample_label = sample_label[:size]
    
    else:
        sample += list(rng.choice(data, size,replace=False))
        sample_label = [get_label(g) for g in sample]
        
    if not output == None:
        with open('samples/'+output+'.npz', 'wb') as f:
                np.savez(f,sample=sample)#, sample_label = sample_label)

    return np.tensor(sample, requires_grad=False), np.tensor(sample_label, requires_grad=False)

class tictactoe():

    def __init__(self, symmetric=True, sample_size=5, design="tceocem tceicem tcedcem", random_sample=False, wins = [-1, 0, 1], reduced = False, loss_fn = 'mse'):

        self.sample_size = sample_size
        self.design = design
        self.random = random_sample
        self.wins = wins
        self.reduced = reduced
        self.epochs = False # used for saving results, turns to True if run_epochs happens
        self.loss_fn = loss_fn

        load_data(self.reduced)

        if loss_fn == 'mse':
            print('using mean squared loss function...')
            self.cost_function = loss_MSE
        elif loss_fn == 'ce':
            print('using cross entropy loss function...')
            self.cost_function = loss_CE
        elif loss_fn == '5q':
            print('using 5q loss function...')
            self.cost_function = loss_5q
        else:
            raise AttributeError(f'loss function: {loss_fn} not available')

        self.symmetric = symmetric

    def random_parameters(self, repetitions=2):
        '''
        sets random parameters for circuit design and amount of repetitions
        '''
        self.repetitions = repetitions
        self.init_params_torch = random_params(repetitions, self.symmetric, self.design)

    def sample_games(self, size):
        '''
        Create random samples with equal amount of wins for X, O and 0
        '''
        self.games_sample , self.label_sample = gen_games_sample(size, wins=self.wins)

    def load_games(self, data_file, size):
        '''
        Loads games and label from file specified by data_file. The first size data points are retained.
        If the file is not found, generate a new sample.
        Currently, load is implemented via numpy.load(file.npz)
        '''
        try:
            with open('samples/'+data_file+'.npz', 'rb') as f:
                            print('Loading data file \n')
                            return np.tensor(np.load(f, allow_pickle = True)['sample'][:size], requires_grad=False)
                            #self.label_sample = np.tensor(np.load(f, allow_pickle = True)['sample_label'][:size*3], requires_grad=False)
        except IOError: 
            print('Data sample not found, creating new one')
            return gen_games_sample(size = 600, wins = self.wins)[0]


    def run_epochs(self, epochs, samplesize_per_step, steps_per_epoch, stepsize, data_file = None):
        """
        Runs Adam training with different epochs
        """
        print('running epochs...')
        print(tabulate([['epochs', epochs], ['stepsize', stepsize], ['symmetric', self.symmetric], ['design', self.design], ['sample size per step', samplesize_per_step], ['steps per epoch', steps_per_epoch], \
            ['wins', self.wins], ['repetitions', self.repetitions], ['corners', corner_qubits], ['edges', edge_qubits], ['center', middle_qubit]]))

        self.epochs = True
        if data_file is None:
            self.batch = gen_games_sample(size = steps_per_epoch*samplesize_per_step, wins=self.wins)[0]
            
        else:
            self.batch = self.load_games(data_file, size = steps_per_epoch*samplesize_per_step)

        self.test_batch = gen_games_sample(size = 600, wins = self.wins)[0]

        rng.shuffle(self.batch)

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
            
            rng.shuffle(self.batch)
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

    def run_lbfgs(self, size, steps, stepsize=0.1, data_file = None, resume = False):
        '''
        Runs qml with torch's lbfgs implementation. Usually converges much quicker than pennylanes standard gradient descent optimizer
        '''
        print('running lbfgs...')
        print(tabulate([['steps', steps], ['stepsize', stepsize], ['symmetric', self.symmetric], ['design', self.design], ['sample size', size], \
            ['random sample', self.random], ['repetitions', self.repetitions], ['wins', self.wins], ['corners', corner_qubits], ['edges', edge_qubits], ['center', middle_qubit]]))
       
        self.epochs = False
        self.interface = 'torch'
        if not resume:
            self.gd_cost = []
            self.theta = self.init_params_torch
            self.total_accuracy = []
            self.training_accuracy = []

        if data_file is None:
            self.batch = gen_games_sample(size = size, wins = self.wins)[0]
        else:
            self.batch = self.load_games(data_file, size) # loads games and labels from file
        


        self.test_batch = gen_games_sample(size = 600, wins = self.wins)[0]

        self.opt = torch.optim.LBFGS([self.theta], lr=stepsize)
        self.stepsize = stepsize
        step_start = 0
        step_end = 0

        def closure():
            self.opt.zero_grad()
            loss = self.cost_function(full_circ_torch, self.theta, self.batch, self.symmetric, self.design)
            loss.backward()
            return loss

        print(f'step 0/{steps} training accuracy:')
        self.training_accuracy.append(self.check_accuracy(check_batch=self.batch))
        print(f'step 0/{steps} test accuracy:')
        self.total_accuracy.append(self.check_accuracy(check_batch=self.test_batch))

        for j in range(1, steps+1):

            cost_temp = self.cost_function(full_circ_torch, self.opt.param_groups[0]['params'][0],self.batch, self.symmetric, self.design)
            
            print(f"step {j-1}/{steps} current cost value: {cost_temp} execution time: {step_end-step_start}s")
            
            step_start = timer()
            self.opt.step(closure)
            step_end = timer()
            # Samples new games for every step
            if self.random:
                self.batch = gen_games_sample(size = self.size, wins = self.wins)[0]

            self.gd_cost.append(cost_temp) 
            self.theta = self.opt.param_groups[0]['params'][0]
            print(f'step {j}/{steps} training accuracy:')
            self.training_accuracy.append(self.check_accuracy(check_batch=self.batch))
            print(f'step {j}/{steps} test accuracy:')
            self.total_accuracy.append(self.check_accuracy(check_batch=self.test_batch))

            self.steps = j
        
        cost_temp = self.cost_function(full_circ_torch ,self.opt.param_groups[0]['params'][0],self.batch, self.symmetric, self.design)
        self.gd_cost.append(cost_temp)
        print(f"final step current cost value: {cost_temp}")

        self.theta = self.opt.param_groups[0]['params'][0]  

    def check_accuracy(self, check_size = None, check_batch = None):
        '''
        checks accuracy of current theta on check_batch OR by sampling check_size amount of games for each win
        '''

        if (check_size is None) and (check_batch is None):
            raise AttributeError('check_size and check_batch cannot both be None')

        if check_batch is None:
            games_check, labels_check = gen_games_sample(check_size, wins=self.wins)
        else: 
            games_check = check_batch
            labels_check = np.tensor([get_label(i) for i in check_batch])

        y_g_raw = []
        y = []

        for i, game in enumerate(games_check):

            y.append(int(labels_check[i]))

            res = get_results(full_circ_torch(game, self.theta, self.symmetric, design=self.design)).detach().numpy()

            y_g_raw.append(res)

        if self.loss_fn == '5q':
            # accuracy for loss_5q
            y_g = []
            for i in y_g_raw:
                if i[1] > 0.5:
                    y_g.append(0)
                elif i[0] > 0.5:
                    y_g.append(1)
                else: 
                    y_g.append(-1)

            self.confusion_matrix = confusion_matrix(y, y_g, normalize='true')
            self.accuracy = self.confusion_matrix.trace()/3

            print('Confusion matrix:')
            print(self.confusion_matrix)

            return self.confusion_matrix

        else:
            # accufacy for loss_MSE and loss_CE
            win_labels = [-1, 0, 1]
            y_g = [win_labels[i.argmax()] for i in y_g_raw]
            self.confusion_matrix = confusion_matrix(y, y_g, normalize='true')
            self.accuracy = self.confusion_matrix.trace()/3

            print('Confusion matrix:')
            print(self.confusion_matrix)

            return self.confusion_matrix
        

    def save(self, name, exec_time=0):
        '''
        saves result of qml as a npy file. Can be analyzed later
        '''
        params_tmp = self.init_params_torch.detach().numpy()
        theta_tmp = self.theta.detach().numpy()

        to_save = {'symmetric': self.symmetric, 'epochs': self.epochs, 'accuracy': self.confusion_matrix,'execution time': exec_time, 'steps': self.steps, 'stepsize': self.stepsize, 'design': self.design, 'interface': self.interface, 'cost function': self.gd_cost, 'sample size': self.sample_size, \
        'initial parameters': params_tmp, 'theta': theta_tmp}
        if self.epochs:
            to_save['epoch cost'] = self.epoch_cost_function
            to_save['epoch accuracy'] = self.epoch_accuracy
            to_save['epoch batch'] = self.batch
            to_save['epoch total accuracy'] = self.epoch_total_accuracy
        else:
            to_save['batch'] = self.batch
            to_save['training accuracy'] = self.training_accuracy
            to_save['total accuracy'] = self.total_accuracy
            
        print('Saving results as {}.npy'.format(name))
        try:
            #np.save('output/'+name, to_save)
            np.save(name, to_save)
            #dd.io.save(name + '.h5', to_save)
        except FileNotFoundError:
            os.makedirs(os.getcwd()+'/output/'+name[::-1].split('/', 1)[1][::-1])
            np.save('output/'+name, to_save)
            #dd.io.save('output/' + name + '.h5', to_save)
