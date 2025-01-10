#!/bin/bash
# read arguments with fallback default values

SYSTEM=${1:-pendulum}
TRAINER=${2:-transformer}
CONFIG_PATH="representation_learning/configs/"$SYSTEM"_config.yaml"
CHECKPOINT_DIR="/common/users/dm1487/tripods/roa_estimation/outputs/"$SYSTEM"/"$TRAINER"/checkpoints/best_model.pth"
OUTPUT_DIR="/common/users/dm1487/tripods/roa_estimation/outputs/"$SYSTEM"/representation_dataset"

# Run visualization
python -m representation_learning.main \
    --system "$SYSTEM" \
    --config "$CONFIG_PATH" \
    --trainer "$TRAINER" \
    --checkpoint "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode create_dataset