import json
import os
import random
import time
from typing import Any, Dict, List, Optional

# External libraries
import openai
from fire import Fire
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from arbench.data_generation.dc.prompt import *
from arbench.utils.inference import inference

load_dotenv()

# Global inference parameters
_default_model = "gpt-4o"
_default_temperature = 0.7
_default_top_p = 0.7

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")



def check_outline_keys(response: Dict) -> bool:
    """Validate outline structure for crime detection."""
    top_level_keys = ["time", "location", "victim", "suspects"]
    victim_keys = ["introduction", "cause_of_death", "murder_weapon"]
    suspect_keys = ["name", "introduction", "relationship", "reason_at_scene", 
                   "suspicion", "motive", "opportunity", "access_to_weapon", 
                   "is_murderer", "evidence"]
    
    # Check top-level keys
    if not all(key in response for key in top_level_keys):
        print(f"Missing keys: {set(top_level_keys) - set(response.keys())}")
        print(response.keys())
        return False
    
    # Check victim keys
    if not all(key in response.get("victim", {}) for key in victim_keys):
        print(f"Missing keys: {set(victim_keys) - set(response.get('victim', {}).keys())}")
        print(response.get("victim", {}).keys())
        return False
    
    # Check suspect keys
    for suspect in response.get("suspects", []):
        if not all(key in suspect for key in suspect_keys):
            print(f"Missing keys: {set(suspect_keys) - set(suspect.keys())}")
            print(suspect.keys())
            return False
    
    return True

def merge_outline(old: Dict, new: Dict) -> Dict:
    # merge victim dict
    if "victim" in new and isinstance(new["victim"], dict):
        old.setdefault("victim", {})
        old["victim"].update(new["victim"])

    # merge suspects list (replace if provided)
    if "suspects" in new and isinstance(new["suspects"], list):
        old["suspects"] = new["suspects"]

    # merge everything else
    for k, v in new.items():
        if k in ["victim", "suspects"]:
            continue
        old[k] = v

    # normalize aliases
    for s in old.get("suspects", []):
        if "suspicion" not in s and "suspicious_points" in s:
            s["suspicion"] = s["suspicious_points"]
            s.pop("suspicious_points", None)

    # ensure victim required keys never disappear
    if "victim" in old:
        old["victim"].setdefault("cause_of_death", "unknown")
        old["victim"].setdefault("murder_weapon", "unknown")

    return old



def convert_init_info(init_info_dict: Dict) -> str:
    """Convert initial information dictionary to formatted string."""
    victim = init_info_dict["victim"]
    suspect_list = [
        f"- suspect {i+1}:\n  - name: {item['name']}\n  - introduction: {item['introduction']}\n"
        for i, item in enumerate(init_info_dict["suspects"]) 
        if item["name"] != "Jason"
    ]
    
    return f"""
time: {init_info_dict["time"]}
location: {init_info_dict["location"]}
victim:
- name: {victim["name"]}
- introduction: {victim["introduction"]}
- cause of death: {victim["cause_of_death"]}
- murder weapon: {victim["murder_weapon"]}
The investigation focuses on four suspects, one of whom is the true murderer:
{"".join(suspect_list)}"""


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
            json.dump(logs, f, indent=4)
        print(f"Outline has been saved to {save_path}")
    except IOError as e:
        print(f"Error saving logs: {e}")


def generate_victim_and_suspects(case_idx: int) -> Dict:
    """Generate victim information and suspects."""
    object_list = [
        "the true murderer",
        "innocent because he/she has both the motive and opportunity but does not have the access to the weapon",
        "innocent because he/she has both the access to the weapon and opportunity but does not have the motive",
        "innocent because he/she has both the access to the weapon and motive but does not have the opportunity"
    ]
    # Generate victim
    task = victim_doc_task
    response = inference([{"role": "user", "content": prompt.format(outline="", task=task)}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
    response = response.choices[0].message.content

################## I ADDED THIS
    response = response.strip()

    # remove ```json fences if present
    if response.startswith("```"):
        response = response.split("```", 2)[1]  # content inside first fence
        response = response.replace("json", "", 1).strip()
##################

    outline = json.loads(response)
    outline["suspects"] = []
    
    # Generate suspects
    for obj in object_list:
        task = suspect_doc_task.format(object=obj)
        response = inference([{"role": "user", "content": prompt.format(outline=json.dumps(outline), task=task)}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
        response = response.choices[0].message.content
        ################### I ADDED THIS
        response = response.strip()

        # remove ```json fences if present
        if response.startswith("```"):
            response = response.split("```", 2)[1]  # content inside first fence
            response = response.replace("json", "", 1).strip()
        ##################
        outline["suspects"].append(json.loads(response))
    
    ############
    for s in outline["suspects"]:
        if "suspicion" not in s and "suspicious_points" in s:
            s["suspicion"] = s["suspicious_points"]
            s.pop('suspicious_points')
    
    if "victim" in outline:
        outline["victim"].setdefault("cause_of_death", "unknown")
        outline["victim"].setdefault("murder_weapon", "unknown")
    ###############
    
    assert check_outline_keys(outline)
    return outline


def enrich_suspect_details(outline: Dict) -> Dict:
    """Enrich suspect details while preserving masked content."""
    expand_key_list = ["reason_at_scene", "suspicion", "motive", "opportunity", "access_to_weapon"]
    mask_list = [None, None, 2, 3, 1]
    
    for i, key in enumerate(expand_key_list):
        mask_content = ""
        if mask_list[i] is not None:
            mask_content = outline["suspects"][mask_list[i]][key]
        
        response = inference([{
            "role": "user", 
            "content": prompt.format(outline=json.dumps(outline), task=enrich_outline_task.format(key=key))
        }], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
        response = response.choices[0].message.content
        ################### I ADDED THIS
        response = response.strip()

        # remove ```json fences if present
        if response.startswith("```"):
            response = response.split("```", 2)[1]  # content inside first fence
            response = response.replace("json", "", 1).strip()
        ##################
        outline = json.loads(response)
        
        if mask_list[i] is not None:
            outline["suspects"][mask_list[i]][key] = mask_content
    
    assert check_outline_keys(outline)
    return outline


def generate_testimonies_and_timelines(outline: Dict) -> Dict:
    """Generate testimonies and timelines for suspects."""
    # Generate testimonies
    model_input = create_testimony_prompt + '\noutline:\n' + json.dumps(outline) + '\noutput:'
    #outline = json.loads(inference([{"role": "user", "content": model_input}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL).choices[0].message.content)
    response = inference(
    [{"role": "user", "content": model_input}],
    model=_default_model,
    json_format=True,
    api_key=API_KEY,
    base_url=BASE_URL
    ).choices[0].message.content

    response = response.strip()
    if response.startswith("```"):
        response = response.split("```", 2)[1]
        response = response.replace("json", "", 1).strip()

    new_outline = json.loads(response)
    outline = merge_outline(outline, new_outline) 
    assert check_outline_keys(outline)
    
    # Generate timelines
    timeline_list = []
    for suspect in outline["suspects"]:
        model_input = (create_timeline_prompt + '\n' + json.dumps(outline) + 
                      f'\n\ncreate the timeline of suspect: {suspect["name"]}\n\noutput:')
        response = inference([{"role": "user", "content": model_input}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
        response = response.choices[0].message.content
        ################### I ADDED THIS
        response = response.strip()

        # remove ```json fences if present
        if response.startswith("```"):
            response = response.split("```", 2)[1]  # content inside first fence
            response = response.replace("json", "", 1).strip()
        ##################
        timeline_list.append(json.loads(response))
    
    for i, timeline in enumerate(timeline_list):
        outline["suspects"][i]["timeline"] = timeline
    
    assert check_outline_keys(outline)
    return outline


def generate_stories_and_questions(outline: Dict) -> Dict:
    """Generate stories and key questions for suspects."""
    # Generate stories
    for idx, suspect in enumerate(outline["suspects"]):
        task = create_story_task.format(suspect=suspect["name"])
        response = inference([{"role": "user", "content": prompt.format(outline=json.dumps(outline), task=task)}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
        response = response.choices[0].message.content
        ################### I ADDED THIS
        response = response.strip()

        # remove ```json fences if present
        if response.startswith("```"):
            response = response.split("```", 2)[1]  # content inside first fence
            response = response.replace("json", "", 1).strip()
        ##################
        outline['suspects'][idx]['story'] = json.loads(response)["story"]
    
    # Generate key questions
    suspects = outline["suspects"]
    outline["key_question"] = [
        f"Can this question deduce that {suspects[0]['name']} had the motive?",
        f"Can this question deduce that {suspects[0]['name']} had the opportunity?",
        f"Can this question deduce that {suspects[0]['name']} had the access to the murder weapon?",
        f"Can this question deduce that {suspects[1]['name']} did not have the access to the murder weapon?",
        f"Can this question deduce that {suspects[2]['name']} did not have the motive",
        f"Can this question deduce that {suspects[3]['name']} did not have the opportunity",
    ]
    
    # Set individual suspect key questions
    suspects[0]["key_question"] = [
        f"Can this question deduce that {suspects[0]['name']} had the motive?",
        f"Can this question deduce that {suspects[0]['name']} had the opportunity?",
        f"Can this question deduce that {suspects[0]['name']} had the access to the murder weapon?"
    ]
    suspects[1]["key_question"] = [f"Can this question deduce that {suspects[1]['name']} did not have the access to the murder weapon?"]
    suspects[2]["key_question"] = [f"Can this question deduce that {suspects[2]['name']} did not have the motive"]
    suspects[3]["key_question"] = [f"Can this question deduce that {suspects[3]['name']} did not have the opportunity"]
    
    return outline


def assign_tasks_and_generate_blank(outline: Dict) -> Dict:
    """Assign tasks to suspects and generate blank suspect."""
    murderer_task = ("You are the true murderer. Your goal is to avoid suspicion and stay undetected. "
                    "Be strategic and deceptive, providing just enough truthful details to seem credible "
                    "while subtly discrediting other suspects. If questioned, remain calm and confident, "
                    "carefully weaving a story that casts doubt elsewhere without revealing your guilt.")
    
    blank_task = ("You have no prior knowledge about the crime. As you converse with the detective, "
                 "try to gather information about the case and then cleverly use what you learn to sow confusion. "
                 "Your goal is to make the detective suspicious of you and believe you might be the murderer.")
    
    standard_role_task = [
        ("Assist the murderer by helping them clear any suspicion and convincingly establish their innocence. "
         "Use persuasive arguments and evidence to support their alibi and divert the detective's focus away from them."),
        ("Play the role of a highly nervous and terrified suspect, frequently offering irrelevant or unrelated details, "
         "and let your fear make you difficult to question effectively"),
        ("Assist the detective in identifying the true murderer. Provide any observations or insights that could help solve the case.")
    ]
    
    random.shuffle(standard_role_task)
    
    # Assign tasks
    outline["suspects"][0]["task"] = murderer_task
    for i in range(1, len(outline["suspects"])):
        outline["suspects"][i]["task"] = standard_role_task[i-1]
    
    # Generate blank suspect
    response = inference([{"role": "user", "content": blank_prompt.format(outline=convert_init_info(outline))}], model=_default_model, json_format=True, api_key=API_KEY, base_url=BASE_URL)
    response = response.choices[0].message.content
    ################### I ADDED THIS
    response = response.strip()

    # remove ```json fences if present
    if response.startswith("```"):
        response = response.split("```", 2)[1]  # content inside first fence
        response = response.replace("json", "", 1).strip()
    ##################
    blank_info = json.loads(response)
    
    outline["suspects"].append({
        "task": blank_task,
        "name": blank_info["name"],
        "introduction": blank_info["introduction"],
        "story": blank_task
    })
    
    return outline


def finalize_outline(outline: Dict, case_idx: int) -> Dict:
    """Finalize outline with initial information and labels."""
    suspect_list = [
        {"name": outline["suspects"][i]['name'], "introduction": outline["suspects"][i]["introduction"]}
        for i in range(len(outline["suspects"]))
    ]
    
    gt = suspect_list[0]
    random.shuffle(suspect_list)
    label = suspect_list.index(gt)
    
    outline["initial_information"] = {
        "time": outline["time"],
        "location": outline["location"],
        "victim": outline["victim"],
        "suspect": suspect_list
    }
    
    outline["label"] = label
    outline["index"] = case_idx + 1
    
    return outline


def main(save_path: str, cnt: int = 1, model: str = "gpt-4o", 
         temperature: float = 0.7, top_p: float = 0.7) -> None:
    """Main function to generate crime detection cases."""
    logs = load_existing_logs(save_path)
    start_idx = len(logs)
    
    # Set global inference parameters for consistency
    global _default_model, _default_temperature, _default_top_p
    _default_model = model
    _default_temperature = temperature
    _default_top_p = top_p
    
    for case_idx in tqdm(range(start_idx, start_idx + cnt)):
        outline = generate_victim_and_suspects(case_idx)
        outline = enrich_suspect_details(outline)
        outline = generate_testimonies_and_timelines(outline)
        outline = generate_stories_and_questions(outline)
        outline = assign_tasks_and_generate_blank(outline)
        outline = finalize_outline(outline, case_idx)
        logs.append(outline)
        save_logs(logs, save_path)


if __name__ == '__main__':
    Fire(main)