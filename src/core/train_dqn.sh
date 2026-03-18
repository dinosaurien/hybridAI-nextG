#!/bin/bash

export PROJECT_ROOT="/home/exposed/Desktop/hybridAI-nextG"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

cd "$PROJECT_ROOT"

echo "Starting RL Training Pipeline via Poetry (NVIDIA/CUDA)..."

poetry run python src/core/control_layer/RL_engines/model_train/dqn_train.py