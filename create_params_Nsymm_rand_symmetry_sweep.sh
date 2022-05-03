#!/bin/bash

declare -a parameters # this array will be stored in the parameters file for slurm

marker=15000 # an index for folders to save results

steps=30 # steps per epoch
statistics=9 # how many runs with identical parameters there will be to acquire statistics
epochs=100 # for running without epochs put -1
points=15 # data points per step
ce="false"
ss=0.008

for reps in 3
do
       for layout in '024613578' '431762058' '358762410' '087521463' '072418653' '521480673' '243608517' '702635184' '142576803' '718523046' '875042361' '760324581' '812506743' '057468132' '631427850' '726180543' '158230674' '613207584' '467180523' '140827563' '473026158' '376480152' '086423715' '381427056' '452378061' '153260847' '386127450' '538604172' '682745103' '645012738' '276140358' 
       do
	    mkdir /scratch/frarzani/pqc_out/epochs/${marker} # this is for cluster
            paramstr="-s false -n ${steps} -p ${points} -l tcemoidcemoidcemoidcemoidcemoid -f /scratch/frarzani/pqc_out/epochs/${marker} -ss ${ss} -re 2 -ep true -epn ${epochs} -ce ${ce} -cg ry -sq ${layout}"
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


printf "%s\n" "${parameters[@]}" > param_file_eps_Nsymm_rand_symmetry.txt

#readarray -t parameters < ./param_file_eps.txt
#echo ${parameters[1]}
