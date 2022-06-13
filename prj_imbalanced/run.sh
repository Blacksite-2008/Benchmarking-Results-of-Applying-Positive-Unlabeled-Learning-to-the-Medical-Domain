#!/bin/bash
#SBATCH --partition=vgpu20
#SBATCH --gres=gpu:1

srun python3.9 train_without_meta.py -num_initial_pos 300 -learning_rate 7 -weight_decay 8 -num_batches 500 -seed 1 -dataset st