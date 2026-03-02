import json
import os
import time
from typing import Any, Dict, List, Optional

import openai
from fire import Fire
from openai import OpenAI
from arbench.utils.inference import inference
from arbench.data_generation.sp.prompt import *
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")


def expand_node(node: Dict, layers_remaining: int, story_tree: Dict) -> None:
    """Recursively expand story tree nodes."""
    if layers_remaining == 0:
        return

    # Generate questions
    prompt = (question_prompt + json.dumps(story_tree) + 
              f"\nNow generate 1-2 questions based on the leaf node: {node['value']}")
    response = inference([{"role": "user", "content": prompt}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
    response = response.choices[0].message.content
    try:
        question_list = json.loads(response)["question"]
    except (json.JSONDecodeError, KeyError):
        return

    node.setdefault('children', [])

    for question in question_list:
        # Generate child node
        prompt = (node_expand_prompt + "\nCurrent story tree:\n" + 
                 json.dumps(story_tree) + f"\nGiven question:{question}\nOutput:")
        
        response = inference([{"role": "user", "content": prompt}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
        response = response.choices[0].message.content
        try:
            child_node = json.loads(response)
            child_node.setdefault('children', [])
            node['children'].append(child_node)
            expand_node(child_node, layers_remaining - 1, story_tree)
        except json.JSONDecodeError:
            continue


def generate_story_tree(root_value: str, layers: int) -> Dict:
    """Generate complete story tree."""
    story_tree = {"value": root_value, "children": []}
    expand_node(story_tree, layers, story_tree)
    return story_tree


def extract_key_questions(node: Dict) -> List[str]:
    """Recursively extract all key questions from story tree."""
    questions = []
    if node.get('key_question'):
        questions.append(node['key_question'])
    
    for child in node.get('children', []):
        questions.extend(extract_key_questions(child))
    
    return questions


def validate_outline(outline: Dict) -> bool:
    """Validate outline structure."""
    required_keys = {"supernatural", "someone_dies", "core_sentence", "story_tree", "bottom", "surface"}
    tree_keys = {"value", "based_question", "key_question", "children"}
    
    def validate_tree(node):
        if not isinstance(node, dict) or not tree_keys.issubset(node.keys()):
            return False
        return all(validate_tree(child) for child in node.get("children", []))
    
    return (required_keys.issubset(outline.keys()) and 
            validate_tree(outline["story_tree"]))


def load_existing_logs(save_path: str) -> List[Dict]:
    """Load existing logs from file."""
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r') as file:
                return json.load(file)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def save_logs(logs: List[Dict], save_path: str) -> None:
    """Save logs to file."""
    try:
        with open(save_path, 'w') as f:
            json.dump(logs, f, indent=2)
    except IOError as e:
        print(f"Error saving logs: {e}")


def generate_outline(supernatural: str, lethal: str) -> Dict:
    """Generate story outline."""
    prompt = core_sentence_prompt.format(supernatural=supernatural, lethal=lethal)
    response = inference([{"role": "user", "content": prompt}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
    response = response.choices[0].message.content
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def add_story_layers(outline: Dict, depth: int = 2) -> Dict:
    """Add bottom and surface layers to outline."""
    # Generate story tree
    root_sentence = outline["core_sentence"]
    story_tree = generate_story_tree(root_sentence, depth)
    
    outline["story_tree"] = story_tree
    outline["story_tree"]["based_question"] = ""
    outline["story_tree"]["key_question"] = ""
    
    # Generate bottom layer
    prompt = "Input:\n" + json.dumps(outline) + "\nBottom:\n"
    response = inference([{"role": "user", "content": outline_prompt + prompt}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
    response = response.choices[0].message.content
    try:
        outline["bottom"] = json.loads(response)["bottom"]
    except (json.JSONDecodeError, KeyError):
        outline["bottom"] = ""
    
    # Generate surface layer
    prompt = "Input: \n" + json.dumps(outline) + '\nSurface:\n'
    response = inference([{"role": "user", "content": surface_prompt + prompt}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
    response = response.choices[0].message.content
    try:
        outline["surface"] = json.loads(response)["surface"]
    except (json.JSONDecodeError, KeyError):
        outline["surface"] = ""
    
    # Extract key questions
    outline["key_question"] = list(set(extract_key_questions(outline["story_tree"])))
    
    return outline


def main(save_path: str, cnt: int = 1, supernatural: bool = False, lethal: bool = True, 
         model: str = "gpt-4o", depth: int = 2, temperature: float = 0.7, top_p: float = 0.7) -> None:
    """Main function to generate story outlines."""
    logs = load_existing_logs(save_path)
    start_idx = len(logs)
    
    # Set global inference parameters for consistency
    global _default_model, _default_temperature, _default_top_p
    _default_model = model
    _default_temperature = temperature
    _default_top_p = top_p
    
    for i in tqdm(range(start_idx, start_idx + cnt)):
        outline = generate_outline(str(supernatural), str(lethal))
        
        if not outline:
            print(f"Failed to generate outline for index {i}")
            continue
        
        outline = add_story_layers(outline, depth)
        outline["index"] = i
        
        if validate_outline(outline):
            print("All keys satisfied")
            logs.append(outline)
            save_logs(logs, save_path)
        else:
            print("Key validation failed, continuing...")


if __name__ == '__main__':
    Fire(main)