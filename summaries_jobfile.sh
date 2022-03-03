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