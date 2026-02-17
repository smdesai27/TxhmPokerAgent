#!/bin/bash
#SBATCH --job-name=ah_stage_d_style_target
#SBATCH --output=logs/stage_d/slurm/%x_%j.out
#SBATCH --error=logs/stage_d/slurm/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "-------------------"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/stage_d/slurm logs/stage_d/targets

module load python/3.11.11-5e66
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_d_fchpa_restore_potmix_p1_20k}"
BASELINE_ITERS="${BASELINE_ITERS:-5000}"
BASELINE_SEEDS="${BASELINE_SEEDS:-5}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-3000}"
BASE_SEED="${BASE_SEED:-42}"
OUTPUT_JSON="${OUTPUT_JSON:-logs/stage_d/targets/solver_style_target_latest.json}"

python3 poker_rl_agent/scripts/build_solver_style_target.py \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" \
  --iterations "${BASELINE_ITERS}" \
  --seeds "${BASELINE_SEEDS}" \
  --episodes_per_seed "${EPISODES_PER_SEED}" \
  --seed "${BASE_SEED}" \
  --output_json "${OUTPUT_JSON}"
