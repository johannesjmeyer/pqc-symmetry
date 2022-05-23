#!/bin/bash

declare -a parameters # this array will be stored in the parameters file for slurm

marker=10000 # an index for folders to save results

steps=30 # steps per epoch
statistics=9 # how many runs with identical parameters there will be to acquire statistics
points=10 # data points per step
ss=0.1

for reps in 3
do
       for layout in 'zcemoidzcemoidzcemoid' 'imeodzcedomzicimcdoez' 'zmieodczmodiecdimzceo' 'czoidmeodzciemdocemiz' 'mczdeioezcmdoiemzicod' 'czmeidoioczmedziomdce' 'dczmeiodziemcocmodize' 'czemdoicizeodmemzoidc' 'imczdoezdiceomoecmdiz' 'zediocmcdzemioeodcizm' 'cdzemoimcodezidocizme' 'oicedmzmezodiceozidcm' 'zecimodicodemzdziemoc' 'zdceiomzoecdmiedozmic' 'oezmdcimdieoczdzcioem' 'ziodemceiomzcdeicdmzo' 'dimczoeidmzcoeimcezdo' 'czedmoieozdimcozcimed' 'czieomdezdoimczocdiem' 'medcziodciemzozmicdeo'
       do
	    mkdir /scratch/frarzani/pqc_out/pdg/${marker} # this is for cluster
            paramstr="-s true -n ${steps} -p ${points} -l t${layout} -f /scratch/frarzani/pqc_out/pdg/${marker} -ss ${ss} -re ${reps} -cg rz"
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


printf "%s\n" "${parameters[@]}" > param_file_pdg_symm_z_rand.txt

#readarray -t parameters < ./param_file_eps.txt
#echo ${parameters[1]}
