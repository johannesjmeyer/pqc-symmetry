#!/bin/sh

LANG=en_US

totruns=8
parruns=4
counter=0 # counts how many runs to launch in parallel


#!/bin/sh

LANG=en_US

counter=0
  

for i in $(seq 1 1 8)
do
    echo "i $i"
    python3 run_ttt.py & 
    
    counter=$((counter+1))
    echo "counter $counter"
    if [ "$counter" -eq "$parruns" ]
    then
        echo "i $i Waiting parallel runs to finish"
        counter=0
        wait
    fi

done

