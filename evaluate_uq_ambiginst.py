import json
import os
from typing import Any
import numpy as np
import string
np.set_printoptions(precision=3, suppress=True)
from jiwer import wer
from src.evaluation import recursive_normalize
from src.common import ambiginst_extract_ans
from src.config import SAMPLE_N

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--answer_key", type=str, required=True)
args = parser.parse_args()


def build_dict(list_of_list):
    item2id = {}
    for curr_list in list_of_list:
        for item in curr_list:
            if item not in item2id:
                item2id[item] = len(item2id)
    return item2id

def compute_entropy(vec: np.ndarray):
    vec = vec + 1e-10
    vec = vec / np.sum(vec)
    entropy = -np.sum(vec * np.log2(vec))
    return entropy

def compute_acc(gts, preds):
    count = 0
    for i in range(len(gts)):
        if gts[i] == preds[i]:
            count += 1
    return count, count / len(preds)

def majority_vote(answers):
    ans2freq = {}
    max_freq = 0
    max_ans = None
    for ans in answers:
        if ans not in ans2freq:
            ans2freq[ans] = 1
        else:
            ans2freq[ans] += 1
        if ans2freq[ans] > max_freq:
            max_ans = ans
            max_freq = ans2freq[ans]
    return max_ans, max_freq

def process_ans(ans: str):
    return ans.strip(string.punctuation)


with open(args.log_path, 'r', encoding='utf-8') as f:
    content = json.load(f)

best_n = SAMPLE_N
num_examples = len(content)

print("--------Uncertainty Quantification-----------")

all_logs = []

for q_idx in range(num_examples):
    curr_log_dict = content[q_idx]
    task_desc = content[q_idx]['orig_instruction']
    if 'isambig' in content[q_idx]:
        inst_ambig = content[q_idx]['isambig']
    else:
        inst_ambig = True

    raw_output_label_sets = curr_log_dict[args.answer_key]
    # Extract answers from model outputs
    raw_output_label_sets = [[ambiginst_extract_ans(x) for x in xx] for xx in raw_output_label_sets]
    # Normalize all model answers
    raw_output_label_sets = recursive_normalize(raw_output_label_sets)

    ans2idx = build_dict(raw_output_label_sets)
    idx2ans = {v: k for k, v in ans2idx.items()}

    gt_ans = content[q_idx]['target']
    orig_q = content[q_idx]['input']

    # Normalize the ground truth to match normalization applied to model outputs
    gt_norm = recursive_normalize([[gt_ans]])[0][0]

    print("Task: ", task_desc)
    print("orig question:\n", orig_q)

    curr_all_rewrite_cots = raw_output_label_sets
    num_rewrite = len(curr_all_rewrite_cots)

    # Handle edge case: no rewrites available
    if num_rewrite == 0 or len(idx2ans) == 0:
        posterior_entropy = 1.0
        data_uncertainty = 0.0
        prop = 0.0

        log_dict = {
            'question': orig_q,
            'answer': gt_ans,
            'answer_norm': gt_norm,
            'rewrite_all_ans': raw_output_label_sets,
            'pred_ans': None,
            'pred_ans_mv_over_rewrites': None,
            'is_correct': False,
            'prop': prop,
            'data_uncertainty': data_uncertainty,
            "total_uncertainty": posterior_entropy,
            'model_uncertainty_list': [1 for _ in range(len(curr_log_dict))],
            "isambig": inst_ambig,
        }
        all_logs.append(log_dict)
        print("No rewrites; skipping uncertainty/accuracy computation for this item.\n")
        continue

    # Build per-rewrite frequency vectors over the discovered answer space
    mv_answers = []
    rewrite_freq_mat = []
    for rewrite_idx in range(num_rewrite):
        rewrite_answer_list = curr_all_rewrite_cots[rewrite_idx]

        rewrite_freq_array = np.zeros(len(idx2ans))
        for ans in rewrite_answer_list:
            rewrite_freq_array[ans2idx[ans]] += 1

        rewrite_freq_array = rewrite_freq_array / best_n
        rewrite_freq_mat.append(rewrite_freq_array)

        mv_ans = majority_vote(rewrite_answer_list)[0]
        mv_answers.append(mv_ans)

    rewrite_freq_mat = np.stack(rewrite_freq_mat, axis=0)

    knowledge_entropy_list = [compute_entropy(rewrite_freq_mat[i]) for i in range(rewrite_freq_mat.shape[0])]
    print("num set: ", len(idx2ans))
    print("GT (norm): ", gt_norm)
    print("MV per rewrite: ", mv_answers)
    knowledge_entropy_list = np.array(knowledge_entropy_list)
    print(knowledge_entropy_list)
    print("knowledge uncertainty", np.mean(knowledge_entropy_list))

    # Posterior over answers = average of rewrite-level distributions
    pred_posterior = np.mean(rewrite_freq_mat, axis=0)
    posterior_entropy = compute_entropy(pred_posterior)

    data_uncertainty = posterior_entropy - np.mean(knowledge_entropy_list)
    prop = data_uncertainty / (posterior_entropy + 1e-6)

    # === Predicted answer (top-1 of posterior) ===
    top1_idx = int(np.argmax(pred_posterior))
    pred_ans = idx2ans[top1_idx]
    # Majority of the per-rewrite majority votes (optional but sometimes insightful)
    mv_over_rewrites = majority_vote(mv_answers)[0]

    # Correctness vs normalized GT
    is_correct = (pred_ans == gt_norm)

    print("total uncertainty:", posterior_entropy)
    print("data uncertainty: ", data_uncertainty)
    print("pred_ans (top-1):", pred_ans)
    print("mv_over_rewrites:", mv_over_rewrites)
    print("is_correct:", is_correct)
    print()

    log_dict = {
        'question': orig_q,
        'answer': gt_ans,                           # original GT
        'answer_norm': gt_norm,                     # normalized GT
        'rewrite_all_ans': raw_output_label_sets,
        'pred_ans': pred_ans,                       # top-1 from posterior
        'pred_ans_mv_over_rewrites': mv_over_rewrites,  # optional, MV across rewrites
        'is_correct': is_correct,                   # boolean correctness
        'prop': prop,
        'data_uncertainty': data_uncertainty,
        "total_uncertainty": posterior_entropy,
        'model_uncertainty_list': knowledge_entropy_list.tolist(),
        'isambig': inst_ambig
    }
    all_logs.append(log_dict)

# Ensure output directory
if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(all_logs, f, indent=4)

# Summary prints
du_list = [x['data_uncertainty'] for x in all_logs]
print("average data uncertainty: ", np.mean(du_list))

# Overall top-1 answer accuracy (only if we produced is_correct)
if len(all_logs) > 0 and 'is_correct' in all_logs[0]:
    acc = np.mean([1.0 if x['is_correct'] else 0.0 for x in all_logs])
    print("overall answer accuracy (top-1 over posterior): ", acc)
