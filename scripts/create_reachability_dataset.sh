#!/bin/bash

SYSTEM=${1:-"pendulum"}
CONFIG_PATH="representation_learning/configs/"$SYSTEM"_config.yaml"
OUTPUT_DIR="outputs/"$SYSTEM"/reachability_dataset/dataset.npz"
python -m representation_learning.classifier.data.create_dataset \
    --system "$SYSTEM" \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \