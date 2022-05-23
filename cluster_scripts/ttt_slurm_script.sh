#!/bin/bash
#
#SBATCH --time=1-00:00:00                    # Runtime in DAYS-HH:MM:SS format
#
#SBATCH --ntasks=20                           # Number of processes
#SBATCH --cpus-per-task=4                    # Number of cores
#SBATCH --mem-per-cpu=3500                    # Memory per cpu in MB (see also --mem)
#
#SBATCH --chdir=/scratch/frarzani         # NOT SURE WHAT TO SET THIS TO, I GUESS I WILL SEE WHEN I CAN LOG IN
#
#SBATCH --mail-user=frarzani@physik.fu-berlin.de # Email to which notifications will be sent
#SBATCH --mail-type=ALL                      # Type of email notification: BEGIN,END,FAIL,ALL

depth=7 # number of repetitions of layers
steps=100 # number of learning steps per epoch
epochs=15 # number of epochs
parruns=5 # how many runs to launch in parallel
counter=0 # counts how many parallel runs have been launched so far
statistics=20 # number of runs for each combination of parameters

echo "\n##############################################"
echo "Circuit depth: $depth"  
echo "Parallel runs: $parruns"  
echo "Toral runs: $datapoints"
echo "Starting runs for symmetric case"
echo "##############################################\n"

# Launches runs for symmetric case
for i in $(seq 1 1 $statistics)
do
    echo "beginning run number $i"
    python3 run_ttt.py -s "true" -n "$steps" -p "6" -l "tcemoid" -f "output/depth_7" -ss "0.008" -sr "true" -re "$depth" -ep "true" -epn $epochs & 
    counter=$((counter+1))
    #echo "counter $counter"
    if [ "$counter" -eq "$parruns" ]
    then
        echo "i $i Waiting parallel runs to finish"
        counter=0
        wait
    fi
done

echo "\n##############################################"
echo "Circuit depth: $depth"  
echo "Parallel runs: $parruns"  
echo "Starting runs for NON symmetric case"
echo "##############################################\n"
# Launches runs for non symmetric case
for i in $(seq 1 1 $statistics)
do
    echo "beginning run number $i"
    python3 run_ttt.py -s "false" -n "$steps" -p "6" -l "tcemoid" -f "output/depth_7" -ss "0.008" -sr "true" -re "$depth" -ep "true" -epn $epochs & 
    counter=$((counter+1))
    #echo "counter $counter"
    if [ "$counter" -eq "$parruns" ]
    then
        echo "i $i Waiting parallel runs to finish"
        counter=0
        wait
    fi
done
