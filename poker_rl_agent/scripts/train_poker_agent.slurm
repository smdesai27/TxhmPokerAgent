#!/bin/bash
#SBATCH --job-name=poker_rl_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# --- detailed job info ---
echo "Job: $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "-------------------"

# --- environment setup ---
# NOTE: Adjust the module load commands or conda activation 
# to match your specific HPC environment.

# Load specific modules found on the system
module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

# --- virtual environment setup ---
# Create a venv to ensure clean dependency management
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate poker_env

# --- run training ---
echo "Starting training..."

# Ensure we are in the correct directory (project root)
# If submitted from the parent folder (antigravity_poker), cd into anitgravity-txhm
if [ -d "anitgravity-txhm" ]; then
    echo "Changing directory to anitgravity-txhm"
    cd anitgravity-txhm
fi

# Ensure output logs folder exists in the project root
mkdir -p logs

echo "Current Directory: $(pwd)"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the training script
python poker_rl_agent/scripts/train.py

echo "Training finished."
