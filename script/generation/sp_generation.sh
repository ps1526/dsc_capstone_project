#!/bin/bash

# Set default values
SAVE_PATH="data/sp_stories.json"
COUNT=1
MODEL="Qwen2.5-32B-Instruct"
SUPERNATURAL=false
LETHAL=true
DEPTH=3
TEMPERATURE=0.7
TOP_P=0.7

# Run the story generator
python3 -m arbench.data_generation.sp.sp_generator \
    --save_path="$SAVE_PATH" \
    --cnt="$COUNT" \
    --model="$MODEL" \
    --supernatural="$SUPERNATURAL" \
    --lethal="$LETHAL" \
    --depth="$DEPTH" \
    --temperature="$TEMPERATURE" \
    --top_p="$TOP_P"