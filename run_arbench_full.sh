#!/bin/bash
#SBATCH --job-name=arbench_clean
#SBATCH --partition=gpu-shared
#SBATCH --account=ddp433
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/arbench_clean_%j.out
#SBATCH --error=logs/arbench_clean_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=== CLEAN AR-Bench + vLLM 0.6.3 BUILD ==="
date
cd "$SLURM_SUBMIT_DIR"

# -------------------------------
# Load modules
# -------------------------------
module load gpu/0.17.3b
module load intel/19.1.3.304/vecir2b
module load cuda/11.2.2

# -------------------------------
# Conda
# -------------------------------
source ~/anaconda3/etc/profile.d/conda.sh
ENV_NAME="arbench_vllm063"

echo "Deleting old environment if it exists..."
conda remove -n $ENV_NAME --all -y || true

echo "Creating fresh environment..."
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

pip install --upgrade pip setuptools wheel

# -------------------------------
# Install Torch FIRST
# -------------------------------
echo "Installing torch..."
pip install \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Verify torch immediately
python - <<EOF
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF

# -------------------------------
# Install HuggingFace stack
# -------------------------------
echo "Installing HF stack..."
pip install \
    numpy==1.26.4 \
    fsspec==2024.6.1 \
    huggingface-hub==0.23.5 \
    tokenizers==0.19.1 \
    transformers==4.43.4 \
    openai==1.40.6

# -------------------------------
# Install vLLM
# -------------------------------
echo "Installing vLLM..."
pip install vllm==0.6.3

pip install fire jq

# Final sanity check
python - <<EOF
import torch, transformers, tokenizers, vllm
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("vllm:", vllm.__version__)
EOF

# -------------------------------
# Launch vLLM
# -------------------------------
HF_MODEL="Qwen/Qwen2.5-7B-Instruct"
SERVED_NAME="qwen2.5-7b-instruct"
GPU_UTIL=0.90

PORT=$(python - <<EOF
import socket
s=socket.socket()
s.bind(('',0))
print(s.getsockname()[1])
s.close()
EOF
)

LOGFILE="logs/vllm_${SLURM_JOB_ID}.log"

echo "Launching vLLM on port $PORT..."

python -u -m vllm.entrypoints.openai.api_server \
    --model "$HF_MODEL" \
    --served-model-name "$SERVED_NAME" \
    --dtype float16 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    > "$LOGFILE" 2>&1 &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

sleep 30

if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "vLLM failed to start"
    tail -n 100 "$LOGFILE"
    exit 1
fi

echo "vLLM running."

# -------------------------------
# AR-Bench Setup
# -------------------------------
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1"

export TEMPERATURE=1.2
export TOP_P=0.99
export REPETITION_PENALTY=1.2
export PRESENCE_PENALTY=0.4
export FREQUENCY_PENALTY=0.4
export MAX_TOKENS=1024

sleep 5

# -------------------------------
# Run evaluators
# -------------------------------
set -x
for evaluator in dc sp gn; do
    if [ "$evaluator" = "gn" ]; then
        python -m arbench.reasoner.gn.gn_evaluator \
            "$SERVED_NAME" zero_shot \
            "data/${evaluator}/test.json" \
            "results/${evaluator}.json" \
            --max_turn 25
    else
        python -m arbench.reasoner.${evaluator}.${evaluator}_evaluator \
            --data_path "data/${evaluator}/test.json" \
            --method zero_shot \
            --output_path "results/${evaluator}.json" \
            --policy_model "$SERVED_NAME" \
            --response_model "$SERVED_NAME" \
            --branch 3 \
            --max_turn 25
    fi
done
set +x

date
echo "=== DONE ==="
