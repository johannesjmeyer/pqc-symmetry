#!/bin/sh

LANG=en_US

parruns=4
counter=0 # counts how many runs to launch in parallel


#!/bin/sh

LANG=en_US

counter=0
  
# Runs different layers on
for i in $(seq 1 1 30)
do
    echo "i $i"
    python3 run_ttt.py -s "true" -n "100" -p "3" -l "tcemoid" -f "vary_depth" -ss "0.008" -sr "true" -re "$i" & 
    python3 run_ttt.py -s "false" -n "100" -p "3" -l "tcemoid" -f "vary_depth" -ss "0.008" -sr "true" -re "$i" &
    counter=$((counter+2))
    echo "counter $counter"
    if [ "$counter" -eq "$parruns" ]
    then
        echo "i $i Waiting parallel runs to finish"
        counter=0
        wait
    fi
done

