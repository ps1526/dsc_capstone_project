# RRMC + blob branch: run experiments (AR-Bench, perturbation + ECE)
# Build: docker build -t rrmc:blob .
# Run:  docker run --rm -e OPENROUTER_API_KEY=... -v $(pwd)/results:/app/results rrmc:blob python run.py blob/blob_dc_gaussian --n_train 5 --n_test 3

FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers (optional; can remove if not using clustering with embeddings)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY config.py .
COPY pipeline.py .
COPY run.py .
COPY configs/ ./configs/
COPY rrmc/ ./rrmc/
COPY scripts/ ./scripts/

# Default: allow override at runtime
ENV PYTHONUNBUFFERED=1
ENV output_dir=/app/results

# Create results dir so bind-mount works
RUN mkdir -p /app/results

# AR-Bench data: mount at runtime as -v "$(pwd)/AR-Bench:/app/AR-Bench"
# or copy into image if you prefer
RUN mkdir -p /app/AR-Bench/data/dc /app/AR-Bench/data/sp /app/AR-Bench/data/gn

CMD ["python", "run.py", "--list"]
