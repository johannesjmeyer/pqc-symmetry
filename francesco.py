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
 
    qml.RX(param[0], wires=8)
    qml.RY(param[1], wires=8)

def outer_layer(param, symm=True):
    '''
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



#@qml.qnode(ttt_dev, interface='torch')
def circuit(game, params, symmetric):
        #params_single = params[0]
        #params_multi = params[1]
        '''
        prepares the all zero comp basis state then iterates through encoding and layers
        input: params, np array of shape r x 2 x 6 
        '''
        # TODO: this should automatically start from the all-zero state in comp basis right?

        ngame = np.pi*0.5*game # normalize entries of game so they are between -pi/2, pi/2

        for r in range(params.shape[0]): # for each of the r repetitions interleave data_enc with row and colum n layers
            #print('r {}'.format(r))

            if symmetric:
                data_encoding(ngame)
                corners(params[r, 0, 0])
                edges(params[r, 0, 1])
                outer_layer(params[r, 0, 5, 0]) 
                corners(params[r, 0, 2])
                edges(params[r, 0, 3])
                center(params[r, 0, 4])


                data_encoding(ngame)
                corners(params[r, 1, 0])
                edges(params[r, 1, 1])
                inner_layer(params[r, 1, 5, 0]) 
                corners(params[r, 1, 2])
                edges(params[r, 1, 3])
                center(params[r, 1, 4])

                data_encoding(ngame)
                corners(params[r, 2, 0])
                edges(params[r, 2, 1])
                diag_layer(params[r, 2, 5, 0]) 
                corners(params[r, 2, 2])
                edges(params[r, 2, 3])
                center(params[r, 2, 4])

            else: 
                data_encoding(ngame)
                corners(params[r, 0:8], False) 
                edges(params[r, 8:16], False) 
                outer_layer(params[r, 16:24], False) 
                corners(params[r, 24:32], False) 
                edges(params[r, 32:40], False) 
                center(params[r, 40:42])


                data_encoding(ngame)
                corners(params[r, 42:50], False) 
                edges(params[r, 50:58], False) 
                inner_layer(params[r, 58:62], False) 
                corners(params[r, 62:70], False) 
                edges(params[r, 70:78], False) 
                center(params[r, 78:80])

                data_encoding(ngame)
                corners(params[r, 80:88], False) 
                edges(params[r, 88:96], False) 
                diag_layer(params[r, 96:100], False) 
                corners(params[r, 100:108], False) 
                edges(params[r, 108:116], False) 
                center(params[r, 116:118]) 

            ### old stuff ###
            #drawer = qml.d
            # raw(data_encoding)
            #print(drawer(ngame))
            #row_layer(params[r,0])
            #drawer = qml.draw(row_layer)
            #print(drawer(params[r,0]))
            #column_layer(params[r,1])

        return qml.expval(qml.PauliZ(8)) # measure one qubit in comp basis



full_circ = qml.QNode(circuit, ttt_dev)
full_circ_torch = qml.QNode(circuit, ttt_dev, interface='torch')
#full_circ_jax = qml.QNode(circuit, ttt_dev, interface='jax')

#full_circ = jax.jit(full_circ_jax)
###################################################
###################################################
###################################################


#game = np.array([[-1, 1, 1], [0, -1, 1], [-1, 0, -1]], requires_grad = False) # just a random game




rng = np.random.default_rng(2021)
#print(params)
#print([params])



#c = full_circ(game, params)

#drawer = qml.draw(full_circ)
#print(drawer(game, params))

#TODO train something on a few labelled data!

############################

def random_params_old(repetitions, symmetric):
    if symmetric:
        return np.array(rng.uniform(low=-1, high=1, size=(repetitions,2,3)), requires_grad = True)
    else:
        return np.wrap_arrays(rng.uniform(low=-1, high=1, size=(repetitions,8+4+4+4+4+1)), requires_grad = True)

def random_params(repetitions, symmetric):
    if symmetric:
        #param_single = torch.tensor(rng.uniform(low=-1, high=1, size=(repetitions,3,5,2)), requires_grad = True)
        #param_multi = torch.tensor(rng.uniform(low=-1, high=1, size=(repetitions,3)), requires_grad = True)
        #return [param_single, param_multi]
        return np.array(rng.uniform(low=-1, high=1, size=(repetitions,3,6,2)), requires_grad = True)
    else:
        return np.array(rng.uniform(low=-1, high=1, size=(repetitions,118)), requires_grad = True)

def random_params_torch(repetitions, symmetric):
    if symmetric:
        #param_single = torch.tensor(rng.uniform(low=-1, high=1, size=(repetitions,3,5,2)), requires_grad = True)
        #param_multi = torch.tensor(rng.uniform(low=-1, high=1, size=(repetitions,3)), requires_grad = True)
        #return [param_single, param_multi]
        return torch.tensor(rng.uniform(low=-1, high=1, size=(repetitions,3,6,2)), requires_grad = True)
    else:
        return torch.tensor(rng.uniform(low=-1, high=1, size=(repetitions,118)), requires_grad = True)

def cost_function(params,game, symmetric):
    return (full_circ(game,params, symmetric)-get_label(game))**2

def cost_function_batch(params,games_batch, symmetric):
    '''
    normalized least squares cost function over batch of data points (games)
    '''
    return sum([(full_circ(g,params, symmetric)-get_label(g))**2 for g in games_batch])/np.shape(games_batch)[0]

def cost_function_torch(params,game, symmetric):
    return (full_circ_torch(game,params, symmetric)-get_label(game))**2

def cost_function_batch_torch(params,games_batch, symmetric):
    '''
    normalized least squares cost function over batch of data points (games)
    '''
    return sum([(full_circ_torch(g,params, symmetric)-get_label(g))**2 for g in games_batch])/np.shape(games_batch)[0]

def gen_games_sample(size, wins=[1, 0, -1]):
    '''
    Generates Tensor with 3*size games that are won equally by X, O and 0
    '''
    games_data,labels = get_data()

    sample = []
    sample_label = []
    
    for j in wins:
        sample += random.sample([a for k, a in enumerate(games_data) if labels[k] == j], size)
        sample_label += size*[j]

    return np.tensor(sample, requires_grad=False), np.tensor(sample_label, requires_grad=False)
class tictactoe():

    def __init__(self, symmetric=True, sample_size=10):
        #self.opt = qml.GradientDescentOptimizer(0.01)
        self.sample_size = sample_size
        self.sample_games(sample_size)
        self.symmetric = symmetric

    def random_parameters(self, size=1, repetitions=2, torch=True):
        if size==1:
            if torch:
                self.init_params = random_params_torch(repetitions, self.symmetric)
            else:
                self.init_params = random_params(repetitions, self.symmetric)
        else:
            # Find best starting paramters
            if torch:
                params_list = [random_params_torch(repetitions, self.symmetric) for i in range(size)]
            else: 
                params_list = [random_params(repetitions, self.symmetric) for i in range(size)]
            cost_list = [cost_function_batch(k,self.games_sample, self.symmetric) for k in params_list]
            self.init_params = params_list[np.argmin(cost_list)]

    def sample_games(self, size):
        # Create random samples with equal amount of wins for X, O and 0
        self.games_sample, self.label_sample = gen_games_sample(size, wins=[-1, 0, 1])

    def run(self, steps, resume = False):
        self.interface = 'pennylane'
        self.opt = qml.GradientDescentOptimizer(0.01)
        if not resume:
            self.gd_cost = []
            self.theta = self.init_params
        for j in range(steps):
            self.theta = self.opt.step(lambda x: cost_function_batch(x, self.games_sample, self.symmetric),self.theta)
            cost_temp = cost_function_batch(self.theta,self.games_sample, self.symmetric)
            print(f"step {j} current cost value: {cost_temp}")
            self.gd_cost.append(cost_temp)   
            self.steps = j     

        print(self.gd_cost)
        print(self.theta)

    def run_torch(self, steps, resume = False):
        self.interface = 'torch'
        if not resume:
            self.gd_cost = []
            self.theta = self.init_params

        self.opt = torch.optim.LBFGS([self.theta], lr=0.1)

        def closure():
            self.opt.zero_grad()
            loss = cost_function_batch_torch(self.theta, self.games_sample, self.symmetric)
            loss.backward()
            return loss
        
        for j in range(steps):
            cost_temp = cost_function_batch_torch(self.opt.param_groups[0]['params'][0],self.games_sample, self.symmetric)
            print(f"step {j} current cost value: {cost_temp}")
            
            self.opt.step(closure)
            #print('step {}'.format(j))
            #print(self.opt.param_groups[0]['params'])

            self.gd_cost.append(cost_temp) 
            self.steps = j
        
        cost_temp = cost_function_batch_torch(self.opt.param_groups[0]['params'][0],self.games_sample, self.symmetric)
        print(f"final step current cost value: {cost_temp}")

        self.theta = self.opt.param_groups[0]['params'][0]  
            
    def check_accuracy(self, check_size = 100):
        # Check what results correspond to which label
        games_check, labels_check = gen_games_sample(check_size)
        results = {-1: {}, 0: {}, 1: {}}
        results_alt = {-1: [], 0: [], 1: []}
        for i, game in enumerate(games_check[:500]):
            if self.interface == 'torch':
                res_device = round(float(full_circ_torch(game, self.theta, self.symmetric)), 3)
            else:
                res_device = round(float(full_circ(game, self.theta, self.symmetric)), 3)
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
        plt.plot(self.gd_cost)
        plt.show()

    def save(self, name):
        to_save = {'symmetric': self.symmetric, 'accuracy': self.accuracy, 'steps': self.steps,  'interface': self.interface, 'cost function': self.gd_cost, 'sample size': self.sample_size, \
        'initial parameters': self.init_params.detach().numpy(), 'sampled games': self.games_sample.numpy(), 'theta': self.theta.detach().numpy()}
        #dd.io.save(name + '.h5', to_save)
        np.save(name, to_save)
# %%
for i in range(20):
    symmetric_run = tictactoe()
    asymetric_run = deepcopy(symmetric_run)
    asymetric_run.symmetric = False

    symmetric_run.random_parameters()
    asymetric_run.random_parameters()

    symmetric_run.run_torch(10)
    asymetric_run.run_torch(10)

    symmetric_run.check_accuracy()
    asymetric_run.check_accuracy()

    symmetric_run.save('symm{}'.format(i))
    asymetric_run.save('asymm{}'.format(i))

# TODO: imply nunmpy save or deepdish H5


# TODO: implement lbfgs optimizer (torch)
# TODO: run on cloud/cluster
# TODO: try different circuits


# TODO: accuracy test?
# TODO: enforce symmetries?
# %%
 