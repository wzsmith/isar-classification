#!/bin/bash

# REQUIRED:
#SBATCH --distribution=cyclic:cyclic
#SBATCH --exclusive

#SBATCH --account="MHPCC96650MZW"
#SBATCH --partition=tesla

#SBATCH --time=025:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
# --------------------
##SBATCH --cpus-per-task=1 --ntasks=8

# Else Python 2.7.5
# module add python36

eval "$(conda shell.bash hook)"

conda activate torch-classification3
python cnn_bilstm.py --train --test weights200.pth