import subprocess
import json
import time
import os

def run_evaluator(cmd_args, output_path):
    """
    Runs AR-Bench evaluator CLI process.
    Returns: output_json
    Raises RuntimeError if evaluator fails or output is empty.
    """
    proc = subprocess.run(cmd_args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Evaluator failed (code={proc.returncode})\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"
        )
    
    # Wait a tiny bit for the output file to be fully written
    for _ in range(5):
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"Output file not created or empty: {output_path}")

    with open(output_path, "r") as f:
        try:
            output = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON decode failed: {e}")
    
    return output

def get_confidence_and_answer(case_path, model, task):
    """
    Runs evaluator on the given case JSON path.
    Returns: evaluator output JSON
    """
    tmp_out = "tmp_eval.json"

    if task == "dc":
        cmd = [
            "python3", "-m", "arbench.reasoner.dc.dc_evaluator",
            "--method", "zero_shot",
            "--data_path", case_path,
            "--output_path", tmp_out,
            "--policy_model", model,
            "--response_model", model,
            "--branch", "3",
            "--max_turn", "25"
        ]
    elif task == "sp":
        cmd = [
            "python3", "-m", "arbench.reasoner.sp.sp_evaluator",
            "--method", "zero_shot",
            "--data_path", case_path,
            "--output_path", tmp_out,
            "--policy_model", model,
            "--response_model", model,
            "--branch", "3",
            "--max_turn", "25"
        ]
    else:
        raise ValueError(f"Unsupported task: {task}")

    return run_evaluator(cmd, tmp_out)
