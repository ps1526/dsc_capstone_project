#!/bin/bash
#SBATCH --job-name=gn_qwen3_vllm
#SBATCH --partition=gpu-shared
#SBATCH --account=ddp433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gn_%j.out
#SBATCH --error=logs/gn_%j.err

set -e

echo "=== GN + HF API job started ==="
date

# --- 1) Setup Environment ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hf3211
cd ~/AR-Bench

# Load keys
export API_KEY="$API_KEY"
export BASE_URL="$BASE_URL"
export POLICY_API_KEY="$POLICY_API_KEY"
export POLICY_BASE_URL="$POLICY_BASE_URL"
export RESPONSE_API_KEY="$RESPONSE_API_KEY"
export RESPONSE_BASE_URL="$RESPONSE_BASE_URL"

echo "--- Environment ---"
echo "Policy URL: $POLICY_BASE_URL"
echo "Response URL: $RESPONSE_BASE_URL"

# --- 2) GN baseline with confidence recording ---
echo "Step 1: Run GN baseline..."
python3 -m arbench.reasoner.gn.gn_evaluator \
  --method zero_shot \
  --model "qwen/qwen3-8b" \
  --data_path data/gn/test.json \
  --output_path results_gn_baseline.json \
  --max_turn 25

echo "Done."
date