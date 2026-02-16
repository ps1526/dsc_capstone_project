python tools/generate_clarification.py \
       --dataset_name ambiginst  \
       --output_path logs/clarification/ambiginst.json \
       --sample --sample_n 2

python forward.py --dataset_name ambiginst \
       --clarification_path logs/clarification/ambiginst.json \
       --output_path logs/forward/ambiginst_forward.json

python evaluate_uq_ambiginst.py \
       --log_path logs/forward/ambiginst_forward.json \
       --output_path logs/uq_eval/ambiginst.json \
       --answer_key clarified_all_ans

python tools/compute_metrics_ambiginst.py
