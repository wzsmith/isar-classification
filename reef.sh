#!/bin/bash

#SBATCH --distribution=cyclic:cyclic
#SBATCH --exclusive

#SBATCH --account="MHPCC96650MZW"
#SBATCH --partition=tesla

#SBATCH --time=025:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
# --------------------
##SBATCH --cpus-per-task=1 --ntasks=8
echo "Job started $(date +%b\ %d\,\ %T)"

eval "$(conda shell.bash hook)"

conda activate torch-classification3
python cnn_bilstm.py --train --test weights5_128.pth


echo "Job finished $(date +%b\ %d\,\ %T)"