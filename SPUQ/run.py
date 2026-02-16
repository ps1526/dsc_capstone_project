from spuq import SPUQ
from llms import LLM
import json

llm = LLM('meta-llama/Llama-3.1-8B-Instruct')
spuq = SPUQ(llm=llm, perturbation='paraphrasing', aggregation='rougeL', n_perturb=3)

with open("squad_v2.json", "r") as f:
    data = json.load(f)

examples = []
for article in data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answers = qa["answers"]
            examples.append({
                "context": context,
                "question": question,
                "answers": answers
            })

predictions = []
references = []
questions = []

for s in examples[:10]:
    context = s["context"]
    question = s["question"]

    messages = [{
        "role": "user",
        "content": f"Answer the question based on the context.\n\nContext: {context}\nQuestion: {question}"
    }]

    result = spuq.run(messages, temperature=0.5)
    pred = result["outputs"]
    
    predictions.append(pred)
    references.append(s["answers"][0]['text'] if s["answers"] else "")
    questions.append(question)

semantic_agreements = []

for pred, ref, q in zip(predictions, references, questions):
    for individual_pred in pred:
        judge_prompt = f"""
Determine whether the following two answers are semantically equivalent - meaning they convey the same information.

Answer 1 (model): {individual_pred}
Answer 2 (reference): {ref}

Respond only with "yes" or "no".
    """

        judge_response = llm.generate([
            {"role": "user", "content": judge_prompt}
        ], temperature=0.5, max_new_tokens=5)

        equivalent = "yes" in judge_response
        semantic_agreements.append(equivalent)

semantic_acc = sum(semantic_agreements) / len(semantic_agreements)
print(f"Semantic equivalence accuracy: {semantic_acc:.2f}")