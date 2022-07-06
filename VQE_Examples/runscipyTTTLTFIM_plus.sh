#!/bin/bash


#SBATCH --job-name=TTT_LTFIM_VQE        # Job name, will show up in squeue outputii
#SBATCH --ntasks=1                   # Number of cores
#SBATCH --nodes=1                      # Ensure that all cores are on one machine
#SBATCH --time=5-20:00:00              # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=100000           # Memory per cpu in MB (see also --mem
#SBATCH --array=0-40
#SBATCH --output=job1_%j.out           # File to which standard out will be written
#SBATCH --error=job1_%j.err            # File to which standard err will be written
#SBATCH --mail-type=BEGIN,END,FAIL                # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=antoniomele.p@gmail.com # Email to which notifications will be sent 
 


##############################################
# INITIALISATION STEPS
##############################################

declare -a parameters
declare -i ll
ll=9
sym="sym"
optn="LBFGSscipy"
#parameters=($(seq 1 1 60))
iteraz=1000000000 #1000
declare -a indexr


declare -a arrp


for p in {1..4..1} 
do
	for nrep in {11..20..1}
	do	
		parameters+=($nrep)
		#indexr+=($nrep)
		arrp+=($p)
	done
done

# main_VQE_ryALL_scipy.py
python main_VQE_scipy_TTT_LTFIM_plus.py ${parameters[$SLURM_ARRAY_TASK_ID]} $iteraz ${arrp[$SLURM_ARRAY_TASK_ID]} $ll $sym $optn #${indexr[$SLURM_ARRAY_TASK_ID]}
