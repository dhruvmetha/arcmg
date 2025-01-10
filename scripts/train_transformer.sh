#!/bin/bash

# Directory containing checkpoints'
SYSTEM=${1:-pendulum}
TRAINER=${2:-transformer}
CONFIG_PATH="representation_learning/configs/"$SYSTEM"_config.yaml"
OUTPUT_DIR="/common/users/dm1487/tripods/roa_estimation/outputs/"$SYSTEM"/"

# Run visualization
python -m representation_learning.main \
    --system "$SYSTEM" \
    --config "$CONFIG_PATH" \
    --trainer "$TRAINER" \
    --output-dir "$OUTPUT_DIR" \
    --mode train