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

echo "=== FULL CLEAN AR-Bench + vLLM BUILD ==="
date
cd "$SLURM_SUBMIT_DIR"

# -------------------------------
# Load cluster modules
# -------------------------------
module purge
module load gpu/0.17.3b
module load intel/19.1.3.304/vecir2b
module load cuda/11.2.2

# -------------------------------
# Conda Fresh Reset
# -------------------------------
source ~/anaconda3/etc/profile.d/conda.sh

ENV_NAME="arbench_clean_env"

echo "Removing old environment (if exists)..."
conda remove -n $ENV_NAME --all -y || true
rm -rf ~/anaconda3/envs/$ENV_NAME || true

echo "Creating new environment..."
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

pip install --upgrade pip setuptools wheel

# -------------------------------
# Clean Scientific Stack
# -------------------------------

# NumPy
pip install --no-cache-dir "numpy<2"

# Torch 
pip install --no-cache-dir \
  torch==2.3.0+cu121 \
  torchvision==0.18.0+cu121 \
  torchaudio==2.3.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Lock transformers 
pip install --no-cache-dir transformers==4.40.2

# Install vLLM normally
pip install --no-cache-dir vllm==0.4.2

# Utilities
pip install fire jq openai

# -------------------------------
# Sanity Check
# -------------------------------
python - <<EOF
import torch, vllm, inspect
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("vllm path:", inspect.getfile(vllm))
print("CUDA available:", torch.cuda.is_available())
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

# -------------------------------
# Wait for vLLM
# -------------------------------
echo "Waiting for vLLM to become ready..."

READY=0
for i in {1..60}; do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        READY=1
        break
    fi
    sleep 5
done

if [ "$READY" -ne 1 ]; then
    echo "vLLM failed to start."
    tail -n 100 "$LOGFILE"
    exit 1
fi

echo "vLLM is running."

# -------------------------------
# Standalone Test
# -------------------------------
echo "Running standalone reasoning test..."

TEST_RESPONSE=$(curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$SERVED_NAME\",
    \"messages\": [
      {\"role\": \"system\", \"content\": \"You are a precise reasoning assistant. Respond with only the final answer.\"},
      {\"role\": \"user\", \"content\": \"If a train travels 60 miles per hour for 2 hours, how far does it travel?\"}
    ],
    \"temperature\": 0.0,
    \"max_tokens\": 200
  }")

# Extract only the model's content
ANSWER=$(echo "$TEST_RESPONSE" | python -c "
import sys, json
data = json.load(sys.stdin)
print(data['choices'][0]['message']['content'])
")

echo "Model answer:"
echo "$ANSWER"

if echo "$ANSWER" | grep -q "120"; then
    echo "✅ Standalone test PASSED."
else
    echo "❌ Standalone test FAILED."
    tail -n 50 "$LOGFILE"
    exit 1
fi

echo "Standalone test complete."

# -------------------------------
# AR-Bench Environment
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
# Run Evaluators
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