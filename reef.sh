#!/bin/bash

#SBATCH --distribution=cyclic:cyclic
#SBATCH --exclusive

#SBATCH --account="MHPCC96650MZW"
#SBATCH --partition=tesla

#SBATCH --time=025:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --output=/work1/scratch/wzsmith/cnn-bilstm-work/out/slurm-%j.out  
# --------------------
##SBATCH --cpus-per-task=1 --ntasks=8

echo "Job started $(date +%b\ %d\,\ %T)"

eval "$(conda shell.bash hook)"
conda activate torch-classification3

python cnn_bilstm.py --train --test weights/weights3_128.pth

echo "Job finished $(date +%b\ %d\,\ %T)"