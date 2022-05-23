#!/bin/bash

declare -a parameters # this array will be stored in the parameters file for slurm

marker=2000 # an index for folders to save results

steps=30 # steps per epoch
statistics=9 # +1 = how many runs with identical parameters there will be to acquire statistics
epochs=100 # for running without epochs put -1
points=15 # data points per step
ss=0.008
ce=false
#for ce in "true" "false"; do
    for reps in $(seq 1 1 3); do # sweeping repetitions of the base layout
        #for ss in $(seq 0.008 0.002 0.01); do # sweep step size
        for p in "cemoid" "cemoidcemoid" "cemoidcemoidcemoid" "cemoidcemoidcemoidcemoid" "cemoidcemoidcemoidcemoidcemoid" ; do # sweeping repetitions of cmoid layout after each data encoding
	    #mkdir ./output/jobarray/epochs/${marker} # this is for laptop
            mkdir /scratch/frarzani/pqc_out/epochs/${marker} # this is for cluster
	        paramstr="-s true -n ${steps} -p ${points} -l t${p} -f /scratch/frarzani/pqc_out/epochs/${marker} -ss ${ss} -re ${reps} -ep true -epn ${epochs} -ce ${ce} -cg ry"
            #printf "%s" "${paramstr}" > ./output/jobarray/no_epochs/${marker}/params.txt # laptop
            printf "%s" "${paramstr}" > /scratch/frarzani/pqc_out/epochs/${marker}/params.txt # cluster
            marker=$((marker+1))
            #echo ${marker}
            #echo ${paramstr}
            for i in $(seq 0 1 $statistics); do 
                parameters+=("${paramstr}") # add as many lines with this set of params as needed to acquire statistics
            done
        done
    done
#done

#for ce in "true" "false"; do
    for reps in $(seq 4 1 5); do # sweeping repetitions of the base layout
        #for ss in $(seq 0.008 0.002 0.01); do # sweep step size
        for p in "cemoid" "cemoidcemoid" "cemoidcemoidcemoid" ; do # sweeping repetitions of cmoid layout after each data encoding
	    #mkdir ./output/jobarray/epochs/${marker} # this is for laptop
            mkdir /scratch/frarzani/pqc_out/epochs/${marker} # this is for cluster
	    paramstr="-s true -n ${steps} -p ${points} -l t${p} -f /scratch/frarzani/pqc_out/epochs/${marker} -ss ${ss} -re ${reps} -ep true -epn ${epochs} -ce ${ce} -cg ry"
            #printf "%s" "${paramstr}" > ./output/jobarray/no_epochs/${marker}/params.txt # laptop
            printf "%s" "${paramstr}" > /scratch/frarzani/pqc_out/epochs/${marker}/params.txt # cluster
            marker=$((marker+1))
            #echo ${marker}
            #echo ${paramstr}
            for i in $(seq 0 1 $statistics); do 
                parameters+=("${paramstr}") # add as many lines with this set of params as needed to acquire statistics
            done
        done
    done
#done

printf "%s\n" "${parameters[@]}" > param_file_eps_lp.txt

#readarray -t parameters < ./param_file_eps.txt
#echo ${parameters[1]}
