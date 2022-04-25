#!/bin/bash

declare -a parameters # this array will be stored in the parameters file for slurm

marker=5000 # an index for folders to save results

steps=30 # steps per epoch
statistics=9 # how many runs with identical parameters there will be to acquire statistics
epochs=100 # for running without epochs put -1
points=15 # data points per step
ce="false"
ss=0.008

for reps in 3
do
       for layout in 'eiomcdmoedicoidmce' 'iemocdeomdicmodcie' 'eocmidomidecoimedc' 'cmodiecioemdcdmeio' 'iodmcedmeoicocidem' 'ecmodidecmioidcmeo' 'ecmidodeicomimdeco' 'emcoidceidomocedmi' 'eimdcocodeimcoemdi' 'eiodmcieomcdoeidmc' 'edoicmecdiomioecdm' 'cdeomideoimcmdcioe' 'doemcimcdoeiceoidm' 'ecimdoicoemdioedcm' 'dcmoeicmodieoeicdm' 'dmceoidmcoeidciemo' 'oeicdmoidecmcmdieo' 'eomcdimdecoiecdoim' 'deomiceoidmcomedci' 'deomcieicomdcmoide' 
       do
	    mkdir /scratch/frarzani/pqc_out/epochs/${marker} # this is for cluster
            paramstr="-s false -n ${steps} -p ${points} -l t${layout} -f /scratch/frarzani/pqc_out/epochs/${marker} -ss ${ss} -re ${reps} -ep true -epn ${epochs} -ce ${ce} -cg ry"
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


printf "%s\n" "${parameters[@]}" > param_file_eps_Nsymm_rand_y.txt

#readarray -t parameters < ./param_file_eps.txt
#echo ${parameters[1]}
