# %%
from tictactoe import *
from circuits_ttt import *
import circuits_ttt

import argparse # parser for command line options
import glob
from timeit import default_timer as timer
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.set_printoptions(precision = 2,linewidth = 200)
# %%

###############################################################
############## parse command line options
###############################################################

def str2bool(v): 
    # To deal with boolean arguments
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        print(v)
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Set options for experiment')

# enforce symmetry
parser.add_argument('-s', "--symmetric", type=str, default = 'true',
                    help='if true, ttt symmetry is enforced')

# length of experiment                  
parser.add_argument('-n', "--num_steps", type=int, default = 50,
                    help='number of gradient descent steps') 

# file containing data points for this experiment                 
parser.add_argument('-d', "--data", type=str, default = 'None',
                    help='file of data points for training') 
                    # we feed the data file for reproducibility and facilitate comparison between symm/asymm

# number of data points infile to use             
parser.add_argument('-p', "--points", type=int, default = 6,
                    help='number of data points to use per step. If epochs are activated, the size of the training set is determined by the product of data points and number of steps.') 

# circuit design          
parser.add_argument('-l', "--layout", type=str, default = 'tcemoid',
                    help='string specifying order of encoding layers: \
                    \n t: encode game -- data encoding\
                    \n c: corners -- 1q trainable \
                    \n e: edges -- 1q trainable\
                    \n m: middle/center -- 1q trainable\
                    \n o: outer layer -- 2q trainable\
                    \n i: inner layer  -- 2q trainable\
                    \nd: diagonal layer -- 2q trainable') 
                

parser.add_argument('-f', "--foldername", type=str, default = 'output',
                    help='filename to save') 

parser.add_argument('-lb', "--lbfgs", type=str, default = 'true', # epochs only implemented with lbfgs
                    help='Use torches lbfgs implementation') 

parser.add_argument('-ss', "--stepsize", type=float, default = 0.008,
                    help='specifies step size of gradient descent optimizer') 

parser.add_argument('-r', "--altresult", type=str, default = 'true',
                    help='if false, encode result in single qubit \n if true, encode result in grid.  ') 

parser.add_argument('-sr', "--samplerandom", type=str, default = 'false', # not implemented for epochs
                    help='if true, pick new random games for each step') 

parser.add_argument('-w', "--wins", default = "0,1,2",
                    help='wins to include in dataset. Seperate numbers with a comma e.g. only including wins for -1 and 0 would look like -1,0. For completely random games set to "R"') 

parser.add_argument('-es', "--excludesymmetry", type=str, default = 'false',
                    help='if true, uses reduced dataset only keeps one game per symmetry group') 
                    
parser.add_argument('-re', "--repetitions", type=int, default = 7,
                    help='how many times to repeat layout') 

parser.add_argument('-ep', "--epochs", type=str, default = 'true',
                    help='uses epochs') 

parser.add_argument('-epn', "--epochssize", type=int, default = 10, # actually specifies how many epochs there will be
                    help='number of epochs') 

parser.add_argument('-ce', "--crossentropy", type=str, default = 'false',
                    help='use cross entropy cost function') 

parser.add_argument('-epm', "--epochmult", type=float, default = 1.,
                    help='how many times to repeat data inside the epoch batch. Can be float.') 
        
parser.add_argument('-cg', "--controlgate", type=str, default = 'x',
                    help='Use different gate: \
                        "x": CNOT \
                        "z": CZ \
                        "rx": CRX \
                        "rz": CRZ \
                        "ry": CRY') 

parser.add_argument('-sq', "--symmetryqubits", type=str, default = '024613578',
                    help='which qubits should be treated symmetrically when symmetric=True\
                        First 4, Second 4 qubits and last qubit are treated symetrically') 

args = parser.parse_args()
###############################################################
############## produce data (mostly useful for debugging)
###############################################################
if args.data == 'None':
    data_name = None
else:
    import os.path
    if not os.path.isfile(args.data): # if the specified data file does not exist
        if str2bool(args.epochs):
            gen_games_sample(args.points, output = args.data, truesize=True)
        else:
            gen_games_sample(args.points, output = args.data) # create data file with specified name and size (# of points)
    data_name = args.data

if args.controlgate == 'rx':
    circuits_ttt.rotation_2q = True
    circuits_ttt.gate_2q = qml.CRX
    circuits_ttt.args_symmetric = {'c': 2, 'e': 2, 'o': 1, 'm': 2, 'i': 1, 'd': 1}
    circuits_ttt.args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4}
elif args.controlgate == 'rz':
    circuits_ttt.rotation_2q = True
    circuits_ttt.gate_2q = qml.CRZ
    circuits_ttt.args_symmetric = {'c': 2, 'e': 2, 'o': 1, 'm': 2, 'i': 1, 'd': 1}
    circuits_ttt.args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4}
elif args.controlgate == 'ry':
    circuits_ttt.rotation_2q = True
    circuits_ttt.gate_2q = qml.CRY
    circuits_ttt.args_symmetric = {'c': 2, 'e': 2, 'o': 1, 'm': 2, 'i': 1, 'd': 1}
    circuits_ttt.args_asymmetric = {'c': 8, 'e': 8, 'o': 8, 'm': 2, 'i': 4, 'd': 4}
elif args.controlgate == 'x':
    circuits_ttt.rotation_2q = False
    circuits_ttt.gate_2q = qml.CNOT
    circuits_ttt.args_symmetric = {'c': 2, 'e': 2, 'o': 0, 'm': 2, 'i': 0, 'd': 0}
    circuits_ttt.args_asymmetric = {'c': 8, 'e': 8, 'o': 0, 'm': 2, 'i': 0, 'd': 0}
elif args.controlgate == 'z':
    circuits_ttt.rotation_2q = False
    circuits_ttt.gate_2q = qml.CZ
    circuits_ttt.args_symmetric = {'c': 2, 'e': 2, 'o': 0, 'm': 2, 'i': 0, 'd': 0}
    circuits_ttt.args_asymmetric = {'c': 8, 'e': 8, 'o': 0, 'm': 2, 'i': 0, 'd': 0}
else:
    raise TypeError
###############################################################
############## run experiment
###############################################################

#filename = args.foldername + f'/r-{args.repetitions}_l-{args.layout}_ss-{args.stepsize}_p-{args.points}_n-{args.num_steps}_s-{args.symmetric}_sr-{args.samplerandom}_wr-{args.winsrandom}-TIME{int(time.time())}'
#filename = args.foldername + '/' + '-'.join(f'{k}={v}' for k, v in vars(args).items()) + f'-TIME{int(time.time())}' + '-'+str(round(np.random.uniform(), 3))
filename = args.foldername + '/' + f'TIME{int(time.time())}' + '-'+str(round(np.random.uniform(), 3))

if 'R' in args.wins:
    wins=[]
else:
    #wins = [int(i) for i in args.wins.split(' ')]
    wins = *map(int, args.wins.split(sep=",")),
    wins = [w-1 for w in wins]

print(f'wins: {wins}\n')

symm_order = [int(i) for i in args.symmetryqubits]
circuits_ttt.corner_qubits = symm_order[:4]
circuits_ttt.edge_qubits = symm_order[4:8]
circuits_ttt.middle_qubit = [symm_order[8]]

start = timer()
exp = tictactoe(symmetric=str2bool(args.symmetric), sample_size=args.points, data_file=data_name, design=args.layout, alt_results=str2bool(args.altresult), \
    random_sample=str2bool(args.samplerandom), wins=wins, reduced=str2bool(args.excludesymmetry), cross_entropy=str2bool(args.crossentropy))

# TODO from here, each each step seems to take forever. I am not sure whether it's my pennylane installation or whether I did something stupid (Fra)
exp.random_parameters(1, repetitions=args.repetitions) # select best of 20 random points as starting point
if str2bool(args.epochs):
    exp.run_epochs(args.epochssize, args.points, args.num_steps, args.stepsize, args.epochmult, data_name)
elif str2bool(args.lbfgs):
    exp.run_lbfgs(args.num_steps, args.stepsize)
else:
    exp.run(args.num_steps)
    
exp.check_accuracy()
end = timer()
exp.save(filename, end - start)
print('Total execution time: {} s'.format(end - start))
