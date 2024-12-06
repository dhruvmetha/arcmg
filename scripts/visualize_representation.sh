#!/bin/bash

# Directory containing checkpoints
CHECKPOINT_DIR="outputs/transformer/20241114_171726/checkpoints/best_model.pth"
CONFIG_PATH="representation_learning/configs/transformer_config.yaml"
TRAINER="transformer"

# Run visualization
python -m representation_learning.main \
    --config "$CONFIG_PATH" \
    --trainer "$TRAINER" \
    --mode visualize \
    --checkpoint "$CHECKPOINT_DIR"