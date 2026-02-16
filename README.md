## Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling

## Introduction

This respository runs code for input clarification ensembling, evaluating how clarification generation can impact accuracy and ece for multiple choice and frq questions. It is based on methods outlined in [this](https://arxiv.org/pdf/2311.08718) paper. 

## Datasets

Multiple Choice Question Answer Datasets:
- [Arc-Challenge](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge)
- [Arc-Easy](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Easy)
- [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)

Open Ended Question Answer Datasets:
- [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA)
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

Instruction Dataset (from paper):
- AmbigInst 

Since these datasets are quite large, a very small subset has been included in this repository for testing 
## Dependencies 

Set up environment and download dependencies.

```bash
mamba env create -f env.yml -n ice
mamba activate ice
```

## Running MCQ
For running code to evaluate mcq, we start with the code below. To specify the dataset, replace ```{dataset}``` with ```arc_chall```,```arc_easy```, or ```openbookqa```. 

```bash
# generate clarifications 
python tools/generate_clarification.py --dataset_name {dataset} --output_path logs/clarification/{dataset}.json --sample --sample_n 2	

# evaluate outputs
python forward.py --dataset_name {dataset} --clarification_path logs/clarification/{dataset}.json --output_path logs/forward/{dataset}_forward.json

# correctness and parsing 
python evaluate_uq_ambiginst.py --log_path logs/forward/{dataset}_forward.json --output_path logs/uq_eval/{dataset}.json --answer_key clarified_all_ans

```
Accuracy and ECE results are computed in ```mcq.ipynb```, found in ```logs\uq_eval\mcq.ipynb```

## Running FRQ 

Similarly for evaluating FRQ, we start with the code below To specify  dataset, replace ```{dataset}``` with ```squad_v2```,```trivia_qa```, or ```truthful_qa```. 

```bash
# generate clarifications 
python tools/generate_clarification.py --dataset_name {dataset} --output_path logs/clarification/{dataset}.json --sample --sample_n 2	

# evaluate outputs
python forward.py --dataset_name {dataset} --clarification_path logs/clarification/{dataset}.json --output_path logs/forward/{dataset}_forward.json
```
Correctness evaluation, accuracy and ece are computed in ```frq.ipynb```, found in ```logs\forward\frq.ipynb```

## Running Instruction Dataset

```bash
python tools/generate_clarification.py --dataset_name ambiginst --output_path logs/clarification/ambiginst.json --sample --sample_n 2	

python forward.py --dataset_name ambiginst --clarification_path logs/clarification/ambiginst.json --output_path logs/forward/ambiginst_forward.json

python evaluate_uq_ambiginst.py --log_path logs/forward/ambiginst_forward.json --output_path logs/uq_eval/ambiginst.json --answer_key clarified_all_ans

python tools/compute_metrics_ambiginst.py
```
Summary Statistics will be displayed 

