import json
from arbench.reasoner.dc.dc_evaluator import DCEvaluator
from arbench.reasoner.sp.sp_evaluator import SPEvaluator
from arbench.reasoner.gn.gn_evaluator import GNEvaluator

def run_task(evaluator_cls, data_path, output_path, max_turn=25):
    evaluator = evaluator_cls(
        policy_model="Qwen3-1.7B-Instruct",
        response_model="Qwen3-1.7B-Instruct",
        branch=3,
        max_turn=max_turn,
        policy_temperature=0.7,
        policy_top_p=0.7,
        response_temperature=0.7,
        response_top_p=0.7
    )

    print(f"Running {evaluator_cls.__name__} on {data_path} ...")
    evaluator.evaluate(data_path, output_path)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    # Run each task family
    run_task(DCEvaluator, "data/dc/test.json", "results_dc.json")
    run_task(SPEvaluator, "data/sp/test.json", "results_sp.json")
    run_task(GNEvaluator, "data/gn/test.json", "results_gn.json")
