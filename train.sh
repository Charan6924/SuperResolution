#!/bin/bash
#SBATCH --job-name=esrgan_train
#SBATCH --account=csds312
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --constraint=gpu2h100
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

uv run TrainingLoop.py

echo "End: $(date)"