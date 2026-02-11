#!/bin/bash
#SBATCH --job-name=poker-train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=1:00:00

python poker_rl_agent/scripts/train.py
