#!/bin/bash

# Directory containing checkpoints
CHECKPOINT_DIR="outputs/20241114_163127/checkpoints/best_model.pth"
TRAINER="distillation"
CONFIG_PATH="representation_learning/configs/"$TRAINER"_config.yaml"

# Run visualization
python -m representation_learning.main \
    --config "$CONFIG_PATH" \
    --trainer "$TRAINER" \
    --mode train
