#!/bin/bash

SYSTEM=${1:-"pendulum"}
CONFIG_PATH="representation_learning/configs/"$SYSTEM"_config.yaml"
OUTPUT_DIR="/common/users/dm1487/tripods/roa_estimation/outputs/"$SYSTEM"/reachability_dataset/dataset.npz"
python -m representation_learning.classifier.data.create_dataset \
    --system "$SYSTEM" \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \