"""
Ollama-only backend for completion_with_backoff.

Assumes you are running `ollama serve` locally and have pulled a model, e.g.:
    ollama pull llama3:8b
"""

import os
import re
import string
import requests
from tenacity import retry, wait_chain, wait_fixed

# Default model tag — change this if needed
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def _ollama_chat_once(model, messages, *, temperature=0.7, max_tokens=None, stop=None,
                      seed=None, top_p=None, top_k=None, num_ctx=None, presence_penalty=None,
                      frequency_penalty=None, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens
    if stop is not None:
        payload["options"]["stop"] = stop
    if seed is not None:
        payload["options"]["seed"] = seed
    if top_p is not None:
        payload["options"]["top_p"] = top_p
    if top_k is not None:
        payload["options"]["top_k"] = top_k
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx
    if presence_penalty is not None:
        payload["options"]["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["options"]["frequency_penalty"] = frequency_penalty

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")

@retry(wait=wait_chain(*[wait_fixed(1) for _ in range(3)] +
                       [wait_fixed(2) for _ in range(2)] +
                       [wait_fixed(3)]))
def completion_with_backoff(**kwargs):
    """
    OpenAI-compatible return shape, but using Ollama.
    Honors 'n' to return multiple choices for sampling.
    """
    model       = os.getenv("LLM_MODEL", LLM_MODEL)
    messages    = kwargs.get("messages", [])
    temperature = kwargs.get("temperature", 0.7)
    max_tokens  = kwargs.get("max_tokens") or kwargs.get("max_tokens_to_sample")
    stop        = kwargs.get("stop")
    base_seed   = kwargs.get("seed")
    top_p       = kwargs.get("top_p")
    top_k       = kwargs.get("top_k")
    num_ctx     = kwargs.get("num_ctx")
    presence_penalty  = kwargs.get("presence_penalty")
    frequency_penalty = kwargs.get("frequency_penalty")
    n           = int(kwargs.get("n", 1))  # <-- important

    # Generate n samples by calling the model n times.
    # If a seed is provided, vary it to avoid identical outputs.
    choices = []
    for i in range(n):
        seed_i = (base_seed + i) if isinstance(base_seed, int) else None
        content = _ollama_chat_once(
            model, messages,
            temperature=temperature, max_tokens=max_tokens, stop=stop,
            seed=seed_i, top_p=top_p, top_k=top_k, num_ctx=num_ctx,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
        )
        choices.append({"message": {"content": content}})

    return {"choices": choices}

# ===== Helper functions preserved from your original common.py ===== #

def majority_vote(answers):
    ans2freq = {}
    max_freq = 0
    max_ans = None
    for ans in answers:
        ans2freq[ans] = ans2freq.get(ans, 0) + 1
        if ans2freq[ans] > max_freq:
            max_ans = ans
            max_freq = ans2freq[ans]
    return max_ans, max_freq


def remove_punctuation(obj):
    translator = str.maketrans('', '', string.punctuation)
    if isinstance(obj, list):
        return [remove_punctuation(x) for x in obj]
    elif isinstance(obj, str):
        return obj.strip().translate(translator).lower()
    else:
        raise NotImplementedError


unk_word_pool = ['unknown', "I don't", "did not", "Not specified", "cannot be determined", "No Answer",
                 "No final answer", "I do not", "N/A", "No information", "It depends on", "I cannot",
                 "I can't ", "I am unable", "don't know", "No answer", "Nobody", "enough information",
                 "specific information", "There is no", "No specific", "not provided", "None", "No character",
                 "did not", "No output", "Cannot answer", "Unavailable", "TBD", "To Be Determined",
                 "I am asking about", "current year", "No translation", "depends on", "has not", 'unclear',
                 "confusion", "incorrect", "not aware of", "invalid", 'no one']


def check_answers(ans_list):
    purified_ans = []
    for ans in ans_list:
        lower_ans = ans.lower()
        if any(unk.lower() in lower_ans for unk in unk_word_pool):
            purified_ans.append('unknown')
        else:
            purified_ans.append(ans)
    return purified_ans


def gsm8k_extract_ans(pred_str):
    ANS_RE = re.compile(r"\$?([0-9,]+)\.?\d*%?")
    pred = re.findall(ANS_RE, pred_str)
    if len(pred) >= 1:
        pred = pred[-1].replace(",", "").replace(" ", "")
        try:
            return int(pred)
        except ValueError:
            return -1
    return -1


def ambiginst_extract_ans(model_ans):
    if "answer is" in model_ans:
        return model_ans.split('answer is')[-1].strip()
    return model_ans
