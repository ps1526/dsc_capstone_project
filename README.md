# SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models

## Introduction

This repository contains code to run Sampling with Perturbation for Uncertainty Quantification, as outlined in this [paper](https://arxiv.org/abs/2403.02509), on active reasoning tasks. There are three methods of perturbations that are tested: paraphrasing, dummy tokens, and system messages. Each of the three methods is tested on 5 datasets: 3 multiple choice question and answer datasets and 2 open ended question and answer datasets. For each dataset, the accuracy and expected calibration error (ECE) is measured for each perturbation type.

Additionally, there are three types of active reasoning tasks: guessing numbers, detective cases, and situation puzzles. All types of perturbations can be run on all types of active reasoning tasks.

## Configuration

1. Set up environment and download dependencies.

```bash
conda create --name spuq_env python=3.10
conda activate spuq_env

conda install pip
pip install -r requirements.txt
```

2. Set API key (OpenRouter)

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

3. Run Experiments

```bash
# Run DC (Detective Cases) - 5 suspects, identify murderer
python run.py fixed_turns --task dc --n-puzzles 5

# Run GN (Guessing Numbers) - Bulls & Cows style game
python run.py fixed_turns --task gn --n-puzzles 5

# Run SP (Situation Puzzles) - Yes/no questions to explain story
python run.py fixed_turns --task sp --n-puzzles 5
```


## Modifications

To run the experiments with different methods, use the following code:

```bash
# Available methods: fixed_turns, self_consistency, semantic_entropy, mi_only, robust_mi, perturbed_mi_ece, ece_only
python run.py self_consistency --task dc --n-puzzles 10
python run.py semantic_entropy --task dc --n-puzzles 10

# Run all methods for comparison
python run.py all_methods --task dc --n-puzzles 10
```

To change the model that is used, edit `configs/base.yaml`:
```yaml
policy_model: qwen/qwen-2.5-7b-instruct  # Recommended (no rate limits)
# policy_model: meta-llama/llama-3.3-70b-instruct:free  # Better but rate limited
```

To change perturbation type, modify line 6 of run.py in the SPUQ folder so that the perturbation parameter is one of the following:
- `paraphrasing`
- `system_message`
- `dummy_token`

To change the aggregation method, modify line 6 of run.py in the SPUQ folder so that the aggregation parameter is one of the following:
- Rouge Score:
    - `rouge1`
    - `rouge2`
    - `rougeL`
- Sentence-BERT embedding cosine similarity:
    - `sbert`
- BERT-Score
    - `bertscore`

As a default, run.py will use `paraphrasing` as the perturbation method and `rougeL` as the aggregation method.

```
spuq = SPUQ(llm=llm, perturbation='paraphrasing', aggregation='rougeL', n_perturb=3)
```

For further modifications:
```bash
# Adjust max turns per episode
python run.py fixed_turns --task gn --n-puzzles 5 --max-turns 50

# List available experiment configs
python run.py --list
```
