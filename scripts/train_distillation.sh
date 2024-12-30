#!/bin/bash

# Directory containing checkpoints
SYSTEM=${1:-pendulum}
TRAINER=${2:-distillation}
CONFIG_PATH="representation_learning/configs/"$SYSTEM"_config.yaml"
OUTPUT_DIR="outputs/"$SYSTEM"/"

# Run visualization
python -m representation_learning.main \
    --system "$SYSTEM" \
    --config "$CONFIG_PATH" \
    --trainer "$TRAINER" \
    --output-dir "$OUTPUT_DIR" \
    --mode train