#!/bin/bash
#SBATCH --job-name=poker_rl_stabilized
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# --- detailed job info ---
echo "Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "-------------------"

# Load modules
module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

# Env Setup
source venv/bin/activate

# Ensure logs dir
mkdir -p logs

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run robust training with default config
# 'default' config includes TargetNet, LayerNorm, Scheduler, GradClip, etc.
python scripts/debug_training.py --config_name stable_v2
