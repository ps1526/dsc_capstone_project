# SPUQ: Perturbation-Based Uncertainty Quantification in Active Reasoning

## Table of Contents

- [Problem Description](#problem-description)
- [Background](#background)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Dependency List](#dependency-list)
- [Dataset Access](#dataset-access)
- [Run Experiments](#run-experiments)
- [Modifications](#modifications)
- [Expected Outputs](#expected-outputs)

## Problem Description
It is well known that large language models have a tendency to hallucinate. In order to be more aware of when a model might hallucinate, uncertainty quantification measures can be used. By measuring the uncertainty of a large language model, people can recalibrate their expectations. Uncertainty quantification gives us insight into when a model is uncertain and likely to provide incorrect answers, compared to when uncertainty is low and the model is likely to provide a correct answer. Uncertainty quantification is straightforward to apply to simple question and answer datasets, but how can it be applied in more complex situations?

Currently, active reasoning tasks are still difficult for large language models. When a model must ask for additional information beyond what is initially provided, it struggles. There are an infinite number of questions that could be asked, and some will allow the model to obtain the correct information much more quickly than others. Uncertainty quantification can be applied to active reasoning tasks to force models to choose the question that results in the least uncertainty. Using uncertainty quantification in active reasoning can solve the problem of inefficiency and incorrect answers.

## Background

This repository contains code to run Sampling with Perturbation for Uncertainty Quantification, as outlined in this [paper](https://arxiv.org/abs/2403.02509), on active reasoning tasks. There are three methods of perturbations that are tested: paraphrasing, dummy tokens, and system messages. Each of the three methods can be tested on 5 question and answer datasets through this repo: 3 multiple choice question and answer datasets and 2 open ended question and answer datasets. For each dataset, the accuracy and expected calibration error (ECE) is measured for each perturbation type.

Additionally, there are three types of active reasoning tasks: guessing numbers, detective cases, and situation puzzles. All types of perturbations can be run on all types of active reasoning tasks.

## Installation

1. Clone the spuq branch of the repository:
   ```bash
   git clone -b spuq --single-branch https://github.com/ps1526/dsc_capstone_project.git
   cd dsc_capstone_project
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Environment Setup

Create an `.env` file in the project root with your API configurations.

To reproduce the results from the report, use OpenRouter and Qwen-2.5-7B-Instruct.

```bash
# API Configuration for Data Generation
API_KEY=your_api_key
BASE_URL=your_base_url

# API Configuration for Evaluation
# Policy model configuration
POLICY_API_KEY=your_policy_api_key
POLICY_BASE_URL=your_policy_base_url

# Response model configuration  
RESPONSE_API_KEY=your_response_api_key
RESPONSE_BASE_URL=your_response_base_url
```

## Dependency List

The following dependencies are required in order to run the experiments.

 ```bash
numpy==1.26.4
sentence_transformers==2.7.0
rouge-score==0.1.2
evaluate==0.4.1
bert-score==0.3.13
openai==1.25.0
 ```

## Dataset Access

The multiple choice and open-ended question and answer datasets can be downloaded at the links below. The active reasoning datasets are included in the data folder of the repository.

Multiple Choice Question Answer Datasets:
- [Arc-Challenge](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge)
- [Arc-Easy](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy)
- [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)

Open Ended Question Answer Datasets:
- [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA)
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

## Run experiments

To run experiments on all multiple choice and open-ended question and answer datasets:

```bash
cd SPUQ
python run.py
```

To run active reasoning experiments:

**Detective Cases**

```bash
python -m arbench.reasoner.dc.dc_evaluator \
    --method="zero_shot" \
    --data_path="data/dc/test.json" \
    --output_path="./zero_shot_dc.json" \
    --policy_model="Qwen2.5-7B-Instruct" \
    --response_model="Qwen2.5-7B-Instruct" \
    --branch="3" \
    --max_turn="25" \
    --policy_temperature="0.7" \
    --policy_top_p="0.7" \
    --response_temperature="0.7" \
    --response_top_p="0.7"
```

**Situation Puzzles**

```bash
python -m arbench.reasoner.sp.sp_evaluator \
    --method="zero_shot" \
    --data_path="data/sp/test.json" \
    --output_path="./zero_shot_sp.json" \
    --policy_model="Qwen2.5-7B-Instruct" \
    --response_model="Qwen2.5-7B-Instruct" \
    --branch="3" \
    --max_turn="25" \
    --policy_temperature="0.7" \
    --policy_top_p="0.7" \
    --response_temperature="0.7" \
    --response_top_p="0.7"
```

**Guessing Numbers**

```bash
python -m arbench.reasoner.gn.gn_evaluator \
    --model="Qwen2.5-32B-Instruct" \
    --method="greedy" \
    --data_path="data/gn/test.json" \
    --output_path="./greedy_gn.json" \
    --max_turn="25"
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

## Expected Outputs

After running the SPUQ method on multiple choice and open-ended question answer datasets, the logs will display the model's accuracy and expected calibration error.

After running the active reasoning experiments on any active reasoning method, a new json file will be created in the root of the repository with the prompts that were provided to the model along with the model's responses to each prompt.
