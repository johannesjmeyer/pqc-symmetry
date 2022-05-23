#!/bin/bash

declare -a parameters

marker=100

steps=20
statistics=5
epochs=50 # for running without epochs

for ce in "true" "false"; do
    for points in $(seq 5 5 20); do
        for ss in $(seq 0.006 0.002 0.01); do # sweep step size
            #mkdir ./output/jobarray/no_epochs/${marker} # this is for laptop
            mkdir /scratch/frarzani/pqc_out/epochs/${marker} # this is for cluster
            #paramstr="-s true -n ${steps} -p ${points} -l tcemoidcemoidcemoid -f output/${marker} -ss ${ss} -sr false -re 3 -ep false -epn ${epochs} -ce ${ce}"
            paramstr="-s false -n ${steps} -p ${points} -l tcemoidcemoidcemoid -f /scratch/frarzani/pqc_out/epochs/${marker} -ss ${ss} -sr true -re 3 -ep true -epn ${epochs} -ce ${ce}"
	    #printf "%s" "${paramstr}" > ./output/jobarray/no_epochs/${marker}/params.txt
            printf "%s" "${paramstr}" >   /scratch/frarzani/pqc_out/epochs/${marker}/params.txt
	    marker=$((marker+1))
            #echo ${marker}
            #echo ${paramstr}
            for i in $(seq 0 1 $statistics); do # 19 is the number of jobs with this parameters
                parameters+=("${paramstr}")
            done
        done
    done
done

printf "%s\n" "${parameters[@]}" > param_file_eps_Nsymm.txt

#readarray -t parameters < ./param_file_no_eps.txt
#echo ${parameters[1]}
