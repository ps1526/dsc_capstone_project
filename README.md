# RRMC

**RRMC (Robust Revision-MI Control)** is the working name for the method described in the Robust-MI Active Inquiry proposal: use **robust self-revision mutual information** as an uncertainty signal, then apply **risk-controlled thresholding** (Clopper–Pearson UCB) to decide **ask vs answer** in interactive tasks like AR-Bench.

In order to run different experiments, switch between branches present in the repo to see how to run different experiments

---

## Branch: `blob` (Weight perturbation + ECE on AR-Bench)

On the **blob** branch, we add an **output-space approximation to weight perturbation** for uncertainty quantification on AR-Bench: because API models do not expose weights, we simulate perturbation via temperature schedules, prompt variants, and optional Gaussian/Laplace noise on embeddings, then combine **self-revision MI** with **Expected Calibration Error (ECE)** for stopping and question selection. See `docs/11_perturbation_ece.md` for details.

### Blob quick start

```bash
# Gaussian perturbation on Detective Cases (with calibration)
python run.py blob/blob_dc_gaussian --n_train 20 --n_test 10

# Laplace perturbation on DC
python run.py blob/blob_dc_laplace --n_train 20 --n_test 10

# Guessing Numbers (diagnostic)
python run.py blob/blob_gn_gaussian --n_puzzles 20

# Full comparison: fixed_turns, robust_mi, perturbed_mi_ece, ece_only
python run.py blob/blob_comparison --n_train 30 --n_test 20
```

### Blob methods and configs

- **Methods:** `perturbed_mi_ece` (MI+ECE), `ece_only` (ECE-only). Configs: `configs/methods/perturbed_mi_ece.yaml`, `configs/methods/ece_only.yaml`.
- **Experiments:** `configs/experiments/blob/` — `blob_dc_gaussian`, `blob_dc_laplace`, `blob_gn_gaussian`, `blob_comparison`.
- **Analysis:** `python scripts/analyze_ece_results.py results analysis_output`

### Requirements (blob)

Same as main; no extra deps. Optional: `matplotlib`, `seaborn` for analysis plots.

---

## Quick Start (main)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (OpenRouter)
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Run Experiments

```bash
# Run DC (Detective Cases) - 5 suspects, identify murderer
python run.py fixed_turns --task dc --n-puzzles 5

# Run GN (Guessing Numbers) - Bulls & Cows style game
python run.py fixed_turns --task gn --n-puzzles 5

# Run SP (Situation Puzzles) - Yes/no questions to explain story
python run.py fixed_turns --task sp --n-puzzles 5
```

### Run with Different Methods

```bash
# Available methods: fixed_turns, self_consistency, semantic_entropy, mi_only, robust_mi, perturbed_mi_ece, ece_only
python run.py self_consistency --task dc --n-puzzles 10
python run.py semantic_entropy --task dc --n-puzzles 10

# Run all methods for comparison
python run.py all_methods --task dc --n-puzzles 10
```

### Configuration Options

```bash
# Adjust max turns per episode
python run.py fixed_turns --task gn --n-puzzles 5 --max-turns 50

# List available experiment configs
python run.py --list
```

### Change Model

Edit `configs/base.yaml`:
```yaml
policy_model: qwen/qwen-2.5-7b-instruct  # Recommended (no rate limits)
# policy_model: meta-llama/llama-3.3-70b-instruct:free  # Better but rate limited
```

---

## Running with Docker

A Docker image is provided for reproducible runs (including blob experiments).

### Build

```bash
docker build -t rrmc:blob .
```

### Run experiments (API key via env)

```bash
# Pass API key at run time
docker run --rm -e OPENROUTER_API_KEY="sk-or-v1-..." \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/AR-Bench:/app/AR-Bench" \
  rrmc:blob python run.py blob/blob_dc_gaussian --n_train 5 --n_test 3
```

### Run with env file (recommended)

Create a `.env` in the repo root (git-ignored) with:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

Then:

```bash
docker run --rm --env-file .env \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/AR-Bench:/app/AR-Bench" \
  rrmc:blob python run.py blob/blob_dc_gaussian --n_train 10 --n_test 5
```

### Mounts

- `results/` — bind-mount so outputs persist on the host.
- `AR-Bench/` — bind-mount if you have AR-Bench data at `./AR-Bench/data/{dc,sp,gn}/`. If not, the image still runs; point config to your data path or copy data into the image.

### List experiments

```bash
docker run --rm rrmc:blob python run.py --list
```

### Using Docker Compose

With a `.env` file in the repo root containing `OPENROUTER_API_KEY=...`:

```bash
docker compose build
docker compose run --rm rrmc python run.py blob/blob_dc_gaussian --n_train 5 --n_test 3
```

Volumes for `results/` and `AR-Bench/` are defined in `docker-compose.yml`.

---

## Results

Results are saved to `results/` as JSON files with:
- Accuracy and average turns per method
- Full episode history (questions, answers, feedback)
- Token usage statistics

On the blob branch, results also include ECE and combined uncertainty metrics; use `scripts/analyze_ece_results.py` for summaries and plots.

---

## Docs

- **00_proposal (implementation-ready spec):** `docs/00_proposal.md`
- **01_literature (curated references):** `docs/01_literature.md`
- **Docs index / reading order:** `docs/README.md`
- **Blob (perturbation + ECE):** `docs/11_perturbation_ece.md`

---

## Secrets / API keys

- Put secrets in a local `.env` file in the repo root (this file is git-ignored).
- Template: `configs/env.example`
- For Docker: use `--env-file .env` or `-e OPENROUTER_API_KEY=...`.
