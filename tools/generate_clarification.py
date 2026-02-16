import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import re
import json
import numpy as np
import copy
import tqdm
import argparse
from src.model_util import ICLModel
from src.prompt_util import load_clarification_system_prompt, load_clarification_user_prompt
from src.data_util import load_data
from src.common import completion_with_backoff

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--sample", action = 'store_true')
parser.add_argument("--sample_n", type = int, required = True)

args = parser.parse_args()


def extract_clarification_ambigqa(model_ans: str):
    lines = model_ans.split('\n')
    extract_list, others = [], []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        # Accept "1. foo", "2) foo", "1)foo", "- foo", or plain text
        m = re.match(r'^(\d+[\.\)]\s*|-)\s*(.+)$', s)
        if m:
            extract_list.append(m.group(2).strip())
        elif s.lower().startswith('clarifications'):
            continue
        else:
            others.append(s)
    return extract_list, others

import re

def extract_clarification_ambiginst(model_output: str):
    # Quick early exit
    if not model_output or "No clarification needed" in model_output:
        return [], []

    # Normalize line endings and strip markdown bold (**...**)
    text = model_output.replace('\r', '')
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # e.g., **Disambiguations:** -> Disambiguations:

    # Locate the Disambiguations section (case-insensitive), grab everything after it
    m = re.search(r'Disambiguations\s*:?\s*(.*)$', text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return [], []

    disambig_block = m.group(1)

    # Extract only the top-level numbered items (one line each)
    # e.g., "1. Rearrange ...", "2. Rearrange ..."
    items = re.findall(r'(?m)^\s*\d+\.\s*(.+?)\s*$', disambig_block)

    # Clean up whitespace and trailing periods (optional)
    items = [it.strip() for it in items if it.strip()]

    return items, []                                


def extract_clarification_nq(model_ans: str):
    lines = model_ans.split('\n')
    extract_list = []    
    others = []
    for line in lines:
        if line.startswith('Rephrase'):
            ext = line[len("Rephrase 1: "):]
            if 'specific' in ext.lower():
                continue
            extract_list.append(ext)
        elif line.startswith('Rephrased question'):
            ext = line[len("Rephrased question 1: "):]
            extract_list.append(ext)
        elif line.startswith('Rephrased'):
            ext = line[len("Rephrased 1: "):]
            extract_list.append(ext)
        elif line.startswith("Question: "):
            break
        else:
            others.append(line)
    return extract_list, others

def extract_clarification_gsm8k(model_output: str):
    extract_list = []
    lines = model_output.split('\n')
    for line in lines:
        if line.startswith("Rephrase "):
            line = line[len("Rephrase 1: "): ]
            extract_list.append(line)
    return extract_list, []

def extract_clarification(model_output, dataset_name):
    return extract_clarification_ambiginst(model_output)
    if dataset_name == 'ambigqa':
        return extract_clarification_ambigqa(model_output)
    elif dataset_name == 'ambiginst':
        return extract_clarification_ambiginst(model_output)
    elif dataset_name == 'nq_open':
        return extract_clarification_nq(model_output)
    elif dataset_name == 'gsm8k':
        return extract_clarification_gsm8k(model_output)


def load_db():
    pos_db_path = 'logs/dataset/ambigqa/ambigqa_train_ambig.json'
    neg_db_path = 'logs/dataset/ambigqa/ambigqa_train_unambig.json'
    with open(pos_db_path, 'r', encoding='utf-8') as f:
        pos_db = json.load(f)
    with open(neg_db_path, 'r', encoding='utf-8') as f:
        neg_db = json.load(f)
    return pos_db, neg_db

def format_query(log_dict, dataset_name, user_prompt, icl_selector):
    orig_inst = log_dict['orig_instruction']
    input_q = log_dict['input']
    prompt_full = user_prompt + '\n\n'
    prompt_full += 'Original Task Instruction: ' + orig_inst + '\n'
    prompt_full += f'Input: ' +  input_q.strip() + '\n\n'
    return prompt_full



def main(args):
    system_prompt = load_clarification_system_prompt(args.dataset_name)
    user_prompt = load_clarification_user_prompt(args.dataset_name)

    if args.dataset_name == 'ambigqa':
        pos_db, neg_db = load_db()
        icl_selector = ICLModel(positive_db = pos_db, negative_db = neg_db)
    else:
        icl_selector = None

    output_path = args.output_path
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("save logs to ", output_path)
    model_index = os.getenv("OPENAI_MODEL", "llama3.1:8b")

    test_data = load_data(args.dataset_name)

    all_results = []
    for idx in tqdm.tqdm(range(len(test_data))):
        case = test_data[idx]
        prompt_full = format_query(case, args.dataset_name, user_prompt, icl_selector)

        max_tokens=512
        if args.sample:
            temperature=1.0
            sample_n = args.sample_n
        else:
            temperature=0
            sample_n = 1

        if len(system_prompt) > 1:
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_full},
            ]
        else:
            messages=[
                {"role": "user", "content": prompt_full},
            ]

        ans_model_list = []
        other_outputs = []
        if args.sample:
            calls = args.sample_n
        else:
            calls = 1

        for _ in range(calls):
            response = completion_with_backoff(
                model=model_index,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,  # always 1; loop handles multiplicity
            )
            #print(response)
            ans_model = response['choices'][0]['message']['content']
            extraction, others = extract_clarification(ans_model, args.dataset_name)
            # If your prompt returns multiple items in one completion,
            # limit to at most 1 (or exactly what you want) per call:
            if args.dataset_name == 'ambigqa':
                extraction = extraction[:1]
            ans_model_list += extraction
            other_outputs.append(others)

        result = copy.deepcopy(case)
        result['orig_inst'] = case['orig_instruction']            
        result['self_clarification'] = ans_model_list
        result['others'] = other_outputs
        all_results.append(result)


    if output_path.endswith(".txt"):
        json_path = output_path.replace('.txt','.json')
    else:
        json_path = output_path
    with open(json_path, 'w', encoding = 'utf-8') as f:
        json.dump(all_results, f, indent = 4)

if __name__ == '__main__':
    main(args)
