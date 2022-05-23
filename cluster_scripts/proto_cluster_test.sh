#!/bin/sh

LANG=en_US

depth=7 # number of repetitions of layers
steps=2 # number of learning steps per epoch
epochs=2 # number of epochs
parruns=2 # how many runs to launch in parallel
counter=0 # counts how many parallel runs have been launched
statistics=2

echo "\n##############################################"
echo "Circuit depth: $depth"  
echo "Parallel runs: $parruns"  
echo "Total runs: $statistics"
echo "Starting runs for symmetric case"
echo "##############################################\n"

# Launches runs for symmetric case
for i in $(seq 1 1 $statistics)
do
    echo "beginning run number $i"
    /usr/bin/time -f "\t%E real,\t%M kb MaxMem" python3 run_ttt.py -s "true" -n "$steps" -p "6" -l "tcemoid" -f "output/depth_7" -ss "0.008" -sr "true" -re "$depth" -ep "true" -epn $epochs & 
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
    /usr/bin/time -f "\t%E real,\t%M kb MaxMem" python3 run_ttt.py -s "false" -n "$steps" -p "6" -l "tcemoid" -f "output/depth_7" -ss "0.008" -sr "true" -re "$depth" -ep "true" -epn $epochs & 
    counter=$((counter+1))
    #echo "counter $counter"
    if [ "$counter" -eq "$parruns" ]
    then
        echo "i $i Waiting parallel runs to finish"
        counter=0
        wait
    fi
done
