#!/bin/bash

declare -a parameters # this array will be stored in the parameters file for slurm

marker=13000 # an index for folders to save results

steps=30 # steps per epoch
statistics=9 # +1 = how many runs with identical parameters there will be to acquire statistics
points=10 # data points per step
ss=0.1

#for ce in "true" "false"; do
    for reps in $(seq 1 1 3); do # sweeping repetitions of the base layout
        #for ss in $(seq 0.008 0.002 0.01); do # sweep step size
        for p in "zcemoid" "zcemoidzcemoid" "zcemoidzcemoidzcemoid" "zcemoidzcemoidzcemoidzcemoid" "zcemoidzcemoidzcemoidzcemoidzcemoid" ; do # sweeping repetitions of cmoid layout after each data encoding
	    #mkdir ./output/jobarray/epochs/${marker} # this is for laptop
            mkdir /scratch/frarzani/pqc_out/pdg/${marker} # this is for cluster
	        paramstr="-s false -n ${steps} -p ${points} -l t${p} -f /scratch/frarzani/pqc_out/pdg/${marker} -ss ${ss} -re ${reps} -cg rz"
            #printf "%s" "${paramstr}" > ./output/jobarray/no_epochs/${marker}/params.txt # laptop
            printf "%s" "${paramstr}" > /scratch/frarzani/pqc_out/pdg/${marker}/params.txt # cluster
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
        for p in "zcemoid" "zcemoidzcemoid" "zcemoidzcemoidzcemoid" ; do # sweeping repetitions of cmoid layout after each data encoding
	    #mkdir ./output/jobarray/epochs/${marker} # this is for laptop
            mkdir /scratch/frarzani/pqc_out/pdg/${marker} # this is for cluster
	        paramstr="-s false -n ${steps} -p ${points} -l t${p} -f /scratch/frarzani/pqc_out/pdg/${marker} -ss ${ss} -re ${reps} -cg rz"
            #printf "%s" "${paramstr}" > ./output/jobarray/no_epochs/${marker}/params.txt # laptop
            printf "%s" "${paramstr}" > /scratch/frarzani/pqc_out/pdg/${marker}/params.txt # cluster
            marker=$((marker+1))
            #echo ${marker}
            #echo ${paramstr}
            for i in $(seq 0 1 $statistics); do 
                parameters+=("${paramstr}") # add as many lines with this set of params as needed to acquire statistics
            done
        done
    done
#done

printf "%s\n" "${parameters[@]}" > param_file_pdg_lp_Nsymm_z.txt

#readarray -t parameters < ./param_file_eps.txt
#echo ${parameters[1]}
