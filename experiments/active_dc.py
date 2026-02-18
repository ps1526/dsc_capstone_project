import argparse
import json
import tempfile
import os
from utils_active import get_confidence_and_answer

def evaluate_confidence(case, model, task):
    """Runs evaluator and returns confidence score (1=correct, 0=wrong)"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump([case], tmp, indent=2)
        tmp_path = tmp.name

    out = get_confidence_and_answer(tmp_path, model, task)
    os.remove(tmp_path)

    # Simplest confidence: success = 1, failure = 0
    confidence = out[0].get("success", 0)
    return confidence

def main(args):
    data = json.load(open(args.data_path))
    all_results = []

    for case in data:
        context = case.copy()

        # 1️⃣ Compute baseline confidence
        baseline_conf = evaluate_confidence(context, args.model, task="dc")
        context["baseline_confidence"] = baseline_conf

        # 2️⃣ Candidate questions
        candidate_questions = [
            f"What is missing in this scenario: {context['description']}?",
            f"What would help decide the outcome?",
            f"Ask a clarifying question for this detective case."
        ]

        best = {"delta": -1, "question": None, "simulated_answer": None, "confidence": None}

        # 3️⃣ Evaluate each candidate
        for q in candidate_questions:
            # a) Simulate NPC answer
            npc_prompt = {
                "description": context["description"],
                "question": q,
                "role": "npc"
            }
            sim_answer_case = context.copy()
            sim_answer_case["extra_query"] = q
            sim_answer_case["npc_role"] = True

            # Get NPC answer
            simulated_answer = evaluate_confidence(sim_answer_case, args.model, task="dc")  # Reuse function
            # Note: in full version you’d generate text answer, but we use success=1 as proxy

            # b) Add simulated Q&A to context
            augmented_case = context.copy()
            augmented_case["qa_history"] = [{"question": q, "answer": simulated_answer}]

            # c) Evaluate confidence with augmented context
            conf = evaluate_confidence(augmented_case, args.model, task="dc")
            delta = conf - baseline_conf

            if delta > best["delta"]:
                best.update({
                    "delta": delta,
                    "question": q,
                    "simulated_answer": simulated_answer,
                    "confidence": conf
                })

        all_results.append({
            "case_id": case["id"],
            "baseline_confidence": baseline_conf,
            "selected_question": best
        })

    json.dump(all_results, open(args.output_path, "w"), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args)
