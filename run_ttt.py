# %%
from tictactoe import *
from circuits_ttt import *

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
                    \n t: encode game \
                    \n c: corners \
                    \n e: edges \
                    \n m: middle/center \
                    \n o: outer layer \
                    \n i: inner layer  \
                    \nd: diagonal layer') 
                

parser.add_argument('-f', "--foldername", type=str, default = 'output',
                    help='filename to save') 

parser.add_argument('-lb', "--lbfgs", type=str, default = 'true',
                    help='Use torches lbfgs implementation') 

parser.add_argument('-ss', "--stepsize", type=float, default = 0.008,
                    help='specifies step size of gradient descent optimizer') 

parser.add_argument('-r', "--altresult", type=str, default = 'true',
                    help='if false, encode result in single qubit \n if true, encode result in grid.  ') 

parser.add_argument('-sr', "--samplerandom", type=str, default = 'false',
                    help='if true, pick new random games for each step') 

parser.add_argument('-wr', "--winsrandom", type=str, default = 'false',
                    help='if true, chooses games randomly without even distribution of wins') 

parser.add_argument('-re', "--repetitions", type=int, default = 7,
                    help='how many times to repeat layout') 

parser.add_argument('-ep', "--epochs", type=str, default = 'true',
                    help='uses epochs') 

parser.add_argument('-epn', "--epochssize", type=int, default = 10,
                    help='number of epochs') 

parser.add_argument('-ce', "--crossentropy", type=str, default = 'false',
                    help='use cross entropy cost function') 

parser.add_argument('-epm', "--epochmult", type=float, default = 1.,
                    help='how many times to repeat data inside the epoch batch. Can be float.') 

args = parser.parse_args()


###############################################################
############## produce data (mostly useful for debugging)
###############################################################
print('1\n')
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

###############################################################
############## run experiment
###############################################################

#filename = args.foldername + f'/r-{args.repetitions}_l-{args.layout}_ss-{args.stepsize}_p-{args.points}_n-{args.num_steps}_s-{args.symmetric}_sr-{args.samplerandom}_wr-{args.winsrandom}-TIME{int(time.time())}'
filename = args.foldername + '/' + '-'.join(f'{k}={v}' for k, v in vars(args).items()) + f'-TIME{int(time.time())}' + '-'+str(np.random.uniform())

start = timer()
exp = tictactoe(symmetric=str2bool(args.symmetric), sample_size=args.points, data_file=data_name, design=args.layout, alt_results=str2bool(args.altresult), \
    random_sample=str2bool(args.samplerandom), random_wins=str2bool(args.winsrandom), cross_entropy=str2bool(args.crossentropy))

# TODO from here, each each step seems to take forever. I am not sure whether it's my pennylane installation or whether I did something stupid (Fra)
exp.random_parameters(1, repetitions=args.repetitions) # select best of 20 random points as starting point
if str2bool(args.epochs):
    exp.run_epochs(args.epochssize, args.points, args.num_steps, args.stepsize, args.epochmult, data_name)
elif str2bool(args.lbfgs):
    exp.run_lbgfs(args.num_steps, args.stepsize)
else:
    exp.run(args.num_steps)
    
exp.check_accuracy()
end = timer()
exp.save(filename, end - start)
print('Total execution time: {} s'.format(end - start))
