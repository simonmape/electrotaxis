#!/bin/bash
#SBATCH --job-name=elec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --partition=short
#SBATCH --output=elec_%A_%a.out
#SBATCH --error=elec_errs_%A_%a.out
#SBATCH --array=1-1331

module purge
module load Anaconda3

conda activate fenicsproject

cd ${SCRATCH}/electrotaxis

#Get parameters for this simulation
task_parameter=$(sed -n ${SLURM_ARRAY_TASK_ID}p electrotaxis_params.txt)

#Get the python dist in this conda environment to run the script
~/.conda/envs/fenicsproject/bin/python electrotaxis_simulation.py "${task_parameter}"
