#!/bin/bash
#SBATCH --job-name=gpt_train
#SBATCH --account=dlw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --constraint=gpu2h100
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

uv run train.py

echo "End: $(date)"