#!/bin/bash

# REQUIRED:
#SBATCH --distribution=cyclic:cyclic
#SBATCH --exclusive

#SBATCH --account="MHPCC96650MZW"
#SBATCH --partition=tesla

#SBATCH --time=000:04:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
# --------------------
##SBATCH --cpus-per-task=1 --ntasks=8

module add singularity/3.5.3

# Else Python 2.7.5
#module add python36

singularity \
    exec \
    --nv \
    --bind ${WORKDIR}:/data \
    ${SLURM_SUBMIT_DIR}/torch.sif \
    python \
    /cnn_bilstm/test.py

#python cnn_bilstm.py --train --test test.py

# maybe useful for getting info about the gpus on target node
#cmds=( \
#"lspci -vnn | grep VGA -A 12" \
#"glxinfo | grep OpenGL" \
#"lshw -numeric -C display" \
#"nvidia-smi" \
#)
#for cmd in "${cmds[@]}"
#do
#    echo
#    echo $cmd
#    eval $cmd
#done