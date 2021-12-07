# %%
import pennylane as qml
from pennylane import numpy as np
from tictactoe import *
import random

ttt_dev = qml.device("default.qubit", wires=9) # the device used to label ttt instances

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
            qml.RX(param, wires=i)

    else:
        for n, i in enumerate(qubits):
            qml.RX(param[n], wires=i)
    
def edges(param, symm=True):

    qubits = [1, 3, 5, 7]

    if symm:
        for i in qubits:
            qml.RX(param, wires=i)

    else:
        for n, i in enumerate(qubits):
            qml.RX(param[n], wires=i)
    
def center(param):
    
    qml.RX(param, wires=8)

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



@qml.qnode(ttt_dev)
def full_circ(game, params, symmetric):
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
                outer_layer(params[r, 0, 0]) 

                data_encoding(ngame)
                inner_layer(params[r, 0, 1]) 

                data_encoding(ngame)
                diag_layer(params[r, 0, 2]) 


                data_encoding(ngame)
                edges(params[r, 1, 0])

                data_encoding(ngame)
                corners(params[r, 1, 1])

                data_encoding(ngame)
                center(params[r, 1, 2])

            else: 
                data_encoding(ngame)
                outer_layer(params[r, 0:8], False) 

                data_encoding(ngame)
                inner_layer(params[r, 8:12], False) 

                data_encoding(ngame)
                diag_layer(params[r, 12:16], False) 


                data_encoding(ngame)
                edges(params[r, 16:20], False)

                data_encoding(ngame)
                corners(params[r, 20:24], False)

                data_encoding(ngame)
                center(params[r, 24])

            ### old stuff ###
            #drawer = qml.d
            # raw(data_encoding)
            #print(drawer(ngame))
            #row_layer(params[r,0])
            #drawer = qml.draw(row_layer)
            #print(drawer(params[r,0]))
            #column_layer(params[r,1])

        return qml.expval(qml.PauliZ(8)) # measure one qubit in comp basis




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

def random_params(repetitions, symmetric):
    if symmetric:
        return np.array(rng.uniform(low=-1, high=1, size=(repetitions,2,3)), requires_grad = True)
    else:
        return np.array(rng.uniform(low=-1, high=1, size=(repetitions,8+4+4+4+4+1)), requires_grad = True)

def cost_function(params,game, symmetric):
    return (full_circ(game,params, symmetric)-get_label(game))**2

def cost_function_batch(params,games_batch, symmetric):
    '''
    normalized least squares cost function over batch of data points (games)
    '''
    return sum([(full_circ(g,params, symmetric)-get_label(g))**2 for g in games_batch])/np.shape(games_batch)[0]

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

    return np.tensor(sample), np.tensor(sample_label)

class tictactoeML():

    def __init__(self, symmetric=True, sample_size=5):
        self.opt = qml.GradientDescentOptimizer(0.01)
        self.sample_games(sample_size)
        self.symmetric = symmetric

    def random_parameters(self, size=1, repetitions=5):
        if size==1:
            self.init_params = random_params(repetitions, self.symmetric)
        else:
            # Find best starting paramters
            params_list = [random_params(repetitions, self.symmetric) for i in range(size)]
            cost_list = [cost_function_batch(k,self.games_sample, self.symmetric) for k in params_list]
            self.init_params = params_list[np.argmin(cost_list)]

    def sample_games(self, size):
        # Create random samples with equal amount of wins for X, O and 0
        self.games_sample, self.label_sample = gen_games_sample(size, wins=[-1, 0, 1])

    def run(self, steps, resume = False):
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
    
    def check_accuracy(self, check_size = 100):
        # Check what results correspond to which label
        games_check, labels_check = gen_games_sample(check_size)
        results = {-1: {}, 0: {}, 1: {}}
        results_alt = {-1: [], 0: [], 1: []}
        for i, game in enumerate(games_check[:500]):
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


# TODO: accuracy test?
# TODO: enforce symmetries?
# %%
