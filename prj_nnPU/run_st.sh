#!/bin/bash
#SBATCH --partition=vgpu20
#SBATCH --gres=gpu:1

srun python3.9 train.py -p exp-st -g 0 -e 200 -seed 1