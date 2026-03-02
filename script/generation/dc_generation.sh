#!/bin/bash

# Set default values
SAVE_PATH="data/dc_custom_cases.json"
COUNT=1
MODEL="Qwen2.5-32B-Instruct"
TEMPERATURE=0.7
TOP_P=0.7

# Run the story generator
python3 -m arbench.data_generation.dc.dc_generator \
    --save_path="$SAVE_PATH" \
    --cnt="$COUNT" \
    --model="$MODEL" \
    --temperature="$TEMPERATURE" \
    --top_p="$TOP_P"