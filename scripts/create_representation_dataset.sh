#!/bin/bash

# Directory containing checkpoints
CHECKPOINT_DIR="outputs/transformer/20241114_171726/checkpoints/best_model.pth"
CONFIG_PATH="representation_learning/configs/transformer_config.yaml"
TRAINER="transformer"
OUTPUT_DIR="/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/arcmg/data/pendulum_lqr/pendulum_representation_dataset_5k"

# Run visualization
python -m representation_learning.main \
    --config "$CONFIG_PATH" \
    --trainer "$TRAINER" \
    --mode create_dataset \
    --checkpoint "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR"
