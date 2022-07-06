#!/bin/bash


#SBATCH --job-name=BPsymXXX       # Job name, will show up in squeue outputi
#SBATCH --ntasks=1                    # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=2-18:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000             # Memory per cpu in MB (see also --mem) 
#SBATCH --array=0-18
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=END                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 

##############################################
# INITIALISATION STEPS
##############################################

declare -a Linp
declare -a Arrp

nrands=1000
ans="nosym"

for L in {4..20..2} #9
do
	for nrep in {55..60..5} #2
	do	
		Linp+=($L)
		Arrp+=($nrep)
	done
done

python main_BP_PQC_XXX.py $nrands ${Linp[$SLURM_ARRAY_TASK_ID]} ${Arrp[$SLURM_ARRAY_TASK_ID]} $ans 

