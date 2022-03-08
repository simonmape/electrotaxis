#!/bin/bash
#SBATCH --job-name=summaries
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --partition=medium
#SBATCH --time=24:00:00

module purge
module load Anaconda3
source activate /data/math-multicellular-struct-devel/hert6124/fenicsproject

cd ${DATA}/electrotaxis

/data/math-multicellular-struct-devel/hert6124/fenicsproject/bin/python extract_summaries.py




#!/bin/bash
#SBATCH --job-name=elec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --partition=short
#SBATCH --output=outputs/summ%A_%a.out
#SBATCH --error=outputs/summ%A_%a.out
#SBATCH --array=1-441

module purge
module load Anaconda3
source activate /data/math-multicellular-struct-devel/hert6124/fenicsproject

cd ${SCRATCH}/electrotaxis

#Get parameters for this simulation
task_parameter=$(sed -n ${SLURM_ARRAY_TASK_ID}p params.txt)

#Get the python dist in this conda environment to run the script
/data/math-multicellular-struct-devel/hert6124/fenicsproject/bin/python summaries.py "${task_parameter}"
