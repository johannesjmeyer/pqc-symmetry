#!/bin/bash
#
#SBATCH --time=1-00:00:00                    # Runtime in DAYS-HH:MM:SS format
#
#SBATCH --ntasks=1                           # Number of processes
#SBATCH --cpus-per-task=8                    # Number of cores
#SBATCH --mem-per-cpu=15000                    # Memory per cpu in MB (see also --mem)
#
#SBATCH --chdir=/home/frarzani/pqc         #    THIS ASSUMES THAT THE SAVE PATH IS GIVEN WITHIN run.py
#
#SBATCH --mail-user=frarzani@physik.fu-berlin.de # Email to which notifications will be sent
#SBATCH --mail-type=ALL                      # Type of email notification: BEGIN,END,FAIL,ALL
#
#SBATCH --array=0-19%10                  # launches an array of X identical jobs id 0..X, each using the allocation above, Y running concurrently

#source /net/opt/spack/buster/spack-dev/opt/linux-debian10-ivybridge/gcc-8.3.0/lmod-8.6.5-vscnztx6cokd2oibqjvon7ykfwf4tumj/lmod/lmod/init/bash
#module use /net/opt/spack/buster/spack-dev/modules/lmod/linux-debian10-x86_64/Core
#module load openmpi/4.1.2-y72mjnh
#source /net/opt/spack/buster/spack-dev/spack/var/spack/environments/buildenv-20220323-1/loads

export MPLCONFIGDIR=../mpl
source ./ttt/bin/activate


depth=7 # number of repetitions of layers
steps=20 # number of learning steps per epoch
epochs=25 # number of epochs
points=30 # number of points to compute gradient

echo " "
echo "##############################################"
echo "Circuit depth: $depth"  
echo "Parallel runs: $parruns"  
echo "Toral runs: $datapoints"
echo "Starting runs for symmetric case"
echo "##############################################"
echo " "

/usr/bin/time -f "\t%E real,\t%M kb MaxMem" python3 -u run_ttt.py -s true -n $steps -p $points -l tcemoid -f output/depth_7 -ss 0.004 -sr true -re $depth -ep true -epn $epochs -ce $ce