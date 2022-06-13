#!/bin/bash
#SBATCH --partition=vgpu40
#SBATCH --gres=gpu:1

srun python3.9 train.py -p exp-ptb -g 0 -e 100 -seed 1