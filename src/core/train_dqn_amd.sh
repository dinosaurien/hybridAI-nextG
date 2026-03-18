#!/bin/bash

# AMD ROCm configuration
export HF_HOME="/opt/rocm_sdk_612/models"
export PROJECT_ROOT="/home/exposed/Desktop/hybridAI-nextG"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

cd "$PROJECT_ROOT"

PYTHON_BIN="/opt/rocm_sdk_612/bin/python3"

echo "Starting RL Training Pipeline..."
$PYTHON_BIN src/core/control_layer/RL_engines/model_train/dqn_train.py