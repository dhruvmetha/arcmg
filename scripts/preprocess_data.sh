#!/bin/bash

SYSTEM_NAME=${1:-pendulum}
CONFIG_PATH="representation_learning/configs/${SYSTEM_NAME}_config.yaml"

python representation_learning/preprocess_data.py --config ${CONFIG_PATH}