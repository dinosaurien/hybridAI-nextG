#!/bin/bash
# Deploy script for AI Core System (Telemetry, Orchestrator, UI)
# This is used for Dino's (personal) custom amd rocm python environment. 
# For standard environments, use deploy.sh together with Poetry (standard Nvidia cuda/torch dependencies).

set -e

# AMD ROCm CONFIGURATION
export HF_HOME="/opt/rocm_sdk_612/models"
export HF_HUB_DISABLE_XET=1
export PROJECT_ROOT="/home/exposed/Desktop/hybridAI-nextG"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:/opt/rocm_sdk_612/rocm_smi/lib:$LD_LIBRARY_PATH"
export PATH="/opt/rocm_sdk_612/bin:/opt/rocm_sdk_612/rocm_smi/bin:$PATH"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
cd "$PROJECT_ROOT"

PYTHON_BIN="/opt/rocm_sdk_612/bin/python3"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6000}"
WEB_PORT="${WEB_PORT:-8080}"
TARGET_METRIC="${TARGET_METRIC:-DRB_PdcpSduDelayDl}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Base arguments
ARGS=(
    "--host" "$HOST"
    "--port" "$PORT"
    "--web-port" "$WEB_PORT"
    "--target-metric" "$TARGET_METRIC"
    "--log-level" "$LOG_LEVEL"
)

# Add optional MiniRocket models if they exist
if [ -f "models/minirocket_xapp_gnb.joblib" ]; then
    ARGS+=("--minirocket-gnb-model" "models/minirocket_xapp_gnb.joblib")
fi

if [ -f "models/minirocket_xapp_ue.joblib" ]; then
    ARGS+=("--minirocket-ue-model" "models/minirocket_xapp_ue.joblib")
fi

# Add DQN model if it exists
if [ -f "models/qnet_offline.pt" ]; then
    ARGS+=("--dqn-model" "models/qnet_offline.pt")
    echo "DQN Model     : models/qnet_offline.pt"
else
    echo "DQN Model     : Not found. Optimizer will run untrained."
fi

echo "=========================================="
echo "Starting AI system on AMD ROCm Environment"
echo "=========================================="
echo "xApp TCP Host : $HOST"
echo "xApp TCP Port : $PORT"
echo "Web UI Port   : $WEB_PORT"
echo "Target Metric : $TARGET_METRIC"
echo "Log Level     : $LOG_LEVEL"
echo "=========================================="

$PYTHON_BIN src/core/main.py "${ARGS[@]}"