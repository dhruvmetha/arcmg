#!/bin/bash

# Directory containing checkpoints
SYSTEM="pendulum"
TRAINER="transformer"
CHECKPOINT_DIR="outputs/"$SYSTEM"/"$TRAINER"/checkpoints/best_model.pth"
CONFIG_PATH="representation_learning/configs/"$SYSTEM"_config.yaml"

# Run visualization
python -m representation_learning.main \
    --system "$SYSTEM" \
    --config "$CONFIG_PATH" \
    --trainer "$TRAINER" \
    --mode visualize \
    --checkpoint "$CHECKPOINT_DIR"