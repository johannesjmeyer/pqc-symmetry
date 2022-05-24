#!/bin/bash
#
#SBATCH --time=1-00:00:00                    # Runtime in DAYS-HH:MM:SS format
#
#SBATCH --ntasks=1                           # Number of processes
#SBATCH --cpus-per-task=8                    # Number of cores
#SBATCH --mem-per-cpu=15000                    # Memory per cpu in MB (see also --mem)
#
#SBATCH --chdir=/home/frarzani/pqc-symmetry         #    THIS ASSUMES THAT THE SAVE PATH IS GIVEN WITHIN run.py
#
#SBATCH --mail-user=frarzani@physik.fu-berlin.de # Email to which notifications will be sent
#SBATCH --mail-type=ALL                      # Type of email notification: BEGIN,END,FAIL,ALL
#
#SBATCH --array=0-87                 # Array of X jobs id 0..X-1, each using the allocation above, Y of which running concurrently
                                         # REMEMBER TO CHANGE X DEPENDING ON THE PARAMETERS TO SWEEP

#source /net/opt/spack/buster/spack-dev/opt/linux-debian10-ivybridge/gcc-8.3.0/lmod-8.6.5-vscnztx6cokd2oibqjvon7ykfwf4tumj/lmod/lmod/init/bash
#module use /net/opt/spack/buster/spack-dev/modules/lmod/linux-debian10-x86_64/Core
#module load openmpi/4.1.2-y72mjnh
#source /net/opt/spack/buster/spack-dev/spack/var/spack/environments/buildenv-20220323-1/loads

export MPLCONFIGDIR=../mpl
source ./ttt/bin/activate


#depth=7 # number of repetitions of layers
#steps=20 # number of learning steps per epoch
#epochs=50 # number of epochs
#points=30 # number of points to compute gradient
#statistics=19 # number of runs for each parameter combination -1

readarray -t parameters < ./param_file_eps_lp.txt

#echo " "
#echo "##############################################"
#echo "Circuit depth: $depth"  
#echo "Parallel runs: $parruns"  
#echo "Toral runs: $datapoints"
#echo "Starting runs for symmetric case"
#echo "##############################################"
#echo " "

/usr/bin/time -f "\t%E real,\t%M kb MaxMem" /home/frarzani/pqc-symmetry/ttt/bin/python3 -u run_ttt.py ${parameters[$SLURM_ARRAY_TASK_ID]}

