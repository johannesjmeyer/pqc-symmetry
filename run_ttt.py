from tictactoe import *
from circuits_ttt import *

import argparse # parser for command line options
import glob


np.set_printoptions(precision = 2,linewidth = 200)

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
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Set options for experiment')

# enforce symmetry
parser.add_argument('-s', "--symmetric", type=str, default = 'false',
                    help='if true, ttt symmetry is enforced')

# length of experiment                  
parser.add_argument('-n', "--num_steps", type=int, default =100,
                    help='number of gradient descent steps') 

# file containing data points for this experiment                 
parser.add_argument('-d', "--data", type=str, default = 'ttt_data_sample',
                    help='file of data points for training') 
                    # we feed the data file for reproducibility and facilitate comparison between symm/asymm

# number of data points in file to use             
parser.add_argument('-p', "--points", type=int, default = 10,
                    help='number of data points to use') 

# circuit design          
parser.add_argument('-l', "--layout", type=str, default = 'tceocem tceicem tcedcem',
                    help='string specifying order of encoding layers: \
                    \n t: encode game \
                    \n c: corners \
                    \n e: edges \
                    \n m: middle/center \
                    \n o: outer layer \
                    \n i: inner layer  \
                    \nd: diagonal layer') 
                

parser.add_argument('-f', "--filename", type=int, default = 10,
                    help='filename to save') 

args = parser.parse_args()


###############################################################
############## produce data (mostly useful for debugging)
###############################################################
print('1\n')
import os.path
if not os.path.isfile(args.data): # if the specified data file does not exist
    gen_games_sample(args.points, wins=[1, 0, -1], output = args.data) # create data file with specified name and size (# of points)

###############################################################
############## run experiment
###############################################################
exp = tictactoe(symmetric=str2bool(args.symmetric), sample_size=args.points, data_file=args.data, design=args.layout)

# TODO from here, each each step seems to take forever. I am not sure whether it's my pennylane installation or whether I did something stupid (Fra)
exp.random_parameters(1) # select best of 20 random points as starting point
exp.run_lbgfs(args.num_steps)
exp.check_accuracy()
exp.save(args.filename)