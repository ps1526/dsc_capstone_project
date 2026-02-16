import json
import datasets
import numpy as np

def load_data(dataset_name):
    data_path = f'logs/dataset/{dataset_name}.json'
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
