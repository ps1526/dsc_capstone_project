import itertools
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import timedelta
from typing import Any, Dict, Generator, List, Union

import nltk
import openai
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from openai import OpenAI
from rouge import Rouge
from together import Together
from tqdm import tqdm

def inference(messages, model, json_format=False):
    if "gpt" in model:
        return inference_openai(messages=messages, model=model, json_format=json_format)
    else:
        return inference_together(messages=messages, model=model, json_format=json_format)


def inference_openai(messages, model = "gpt-3.5-turbo", json_format=False):

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
    if json_format:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7, # fixed
            top_p=0.7, # fixed
            response_format={"type": "json_object"}
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7, # fixed
            top_p=0.7, # fixed
            )
    
    return response

def inference_together(messages, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", json_format=False):
    client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

    if json_format:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7, # fixed
            top_p=0.7, # fixed
            response_format={"type": "json_object"}
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7, # fixed
            top_p=0.7, # fixed
            )
    return response


def get_embedding(text, model="text-embedding-3-large"):
    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding


def cosine_similarity(vector1, vector2):
    # 计算两个向量的余弦相似度
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def bleu_score(prediction, ground_truth):
    # 分词
    reference = [nltk.word_tokenize(ground_truth)]
    candidate = nltk.word_tokenize(prediction)

    # 使用 SmoothingFunction 来避免 BLEU 分数为零
    smoothie = SmoothingFunction().method4

    # 计算 BLEU 分数
    bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    return bleu_score


def contains_number(input_string):
    return bool(re.search(r"\d", input_string))


def place_keypoints(keypoints):
    text = ""
    for i, key in enumerate(keypoints):
        text += f"{i+1}. {key}\n"
    return text


def parse_keypoints(input_string):
    # 使用正则表达式提取跟在 "Hit point" 后面的数字
    matches = re.findall(r"Hit point: ([\d, ]+)", input_string)

    # 将提取的数字字符串分割并存入列表
    numbers = []
    if matches:
        try:
            numbers = [int(num.strip()) for num in matches[0].split(",") if num.strip()]
        except:
            print(f" ==> input_string: {input_string}, matches: {matches}")
            raise ValueError
    return numbers


def parse_match_info(text):
    # 正则表达式匹配 "Match point:" 后的内容，允许任意字符，直到 "Match count:"
    match_pattern = r"Match point:\s*([\d,\s]*?),\s*Match count:\s*(\d+)"

    # 使用 re.search 查找匹配的部分
    match = re.search(match_pattern, text)

    if match:
        # 提取 "Match point:" 和 "Match count:" 后的内容
        match_point = match.group(1).strip()
        match_count = int(match.group(2).strip())
        return match_point, match_count
    else:
        return None, 0


def extract_questioner_respond(text):
    return "question:" in text.lower() or "answer:" in text.lower()


class NotQuestionError(Exception):
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "There is not a question"


import random


class NotNumberError(Exception):
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "There is not a number"


def generate_unique_four_digit_number():
    digits = list(range(10))
    random.shuffle(digits)
    # 保证第一位数字不为0
    if digits[0] == 0:
        # 如果第一位是0，与后面任意一位交换
        for i in range(1, 10):
            if digits[i] != 0:
                digits[0], digits[i] = digits[i], digits[0]
                break
    # 取前四位组成四位数
    number = digits[:4]
    # four_digit_number = int(''.join(map(str, number)))
    return number


def extract_and_convert_guess(s):
    matches = re.findall(r"\d{4}", s)

    if matches:
        # 提取匹配的数字字符串
        number_string = matches[-1]

        # 将数字字符串转换为数字列表
        number_list = [int(char) for char in number_string]

        return number_list
    else:
        # 如果没有匹配到 'Guess: ' 后面的数字，返回空列表
        return []


def compare_guess(list1, list2):
    if len(list1) != len(list2):
        raise ValueError

    # 计算相同位置相同的元素数量
    same_position_count = sum(a == b for a, b in zip(list1, list2))

    # 统计两个列表中所有元素的频率
    list1_counts = Counter(list1)
    list2_counts = Counter(list2)

    # 计算两个 Counter 的交集，得到相同元素的总数量（包括相同位置和不同位置）
    same_elements = sum((list1_counts & list2_counts).values())

    # 相同元素但不同位置的数量 = 总相同元素数量 - 相同位置相同的数量
    same_element_different_position_count = same_elements - same_position_count

    # 计算最终得分
    value = same_position_count * 2 + same_element_different_position_count

    return same_position_count, same_element_different_position_count, value
