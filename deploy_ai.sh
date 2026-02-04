#!/bin/bash
# Deploy script for AI system (AMD ROCm Optimized)
# Usage: ./deploy_ai.sh [options]

set -e
# --- START AMD ROCm CONFIGURATION (for Dino hehe) ---
export HF_HOME="/opt/rocm_sdk_612/models"
export PROJECT_ROOT="/home/exposed/Desktop/hybridAI-nextG"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$PROJECT_ROOT/src/demo:$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"

# Get the directory where the script is located
#SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6000}"
TARGET_METRIC="${TARGET_METRIC:-DRB_PdcpSduDelayDl}"
TARGET_VALUE="${TARGET_VALUE:-30.0}"
CELLS="${CELLS:-CELL_001}"
SLICES="${SLICES:-SLICE_A}"
STEPS="${STEPS:-1000}"
LOG_LEVEL="${LOG_LEVEL:-8,2,9}"

# Loss plotting options
SAVE_LOSS_HISTORY="${SAVE_LOSS_HISTORY:-true}"
PLOT_LOSS="${PLOT_LOSS:-false}"
STABLE_LOSS_FILE="${STABLE_LOSS_FILE:-models/loss_history_stable.json}"
UNSTABLE_LOSS_FILE="${UNSTABLE_LOSS_FILE:-models/loss_history_unstable.json}"
LOSS_PLOT_OUTPUT="${LOSS_PLOT_OUTPUT:-loss_plot.png}"

# --- LLM SETTINGS ---
# Set USE_LLM to true by default sinced im using a gpt-2 biased model
USE_LLM="${USE_LLM:-true}"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed. Please install Poetry first."
    exit 1
fi

ARGS=(
    "--host" "$HOST"
    "--port" "$PORT"
    "--target-metric" "$TARGET_METRIC"
    "--target-value" "$TARGET_VALUE"
    "--cells" $CELLS
    "--slices" $SLICES
    "--steps" "$STEPS"
    "--log-level" "$LOG_LEVEL"
)

# Add optional MiniRocket models if they exist
if [ -f "models/minirocket_xapp_gnb.joblib" ]; then
    ARGS+=("--minirocket-gnb-model" "models/minirocket_xapp_gnb.joblib")
fi

if [ -f "models/minirocket_xapp_ue.joblib" ]; then
    ARGS+=("--minirocket-ue-model" "models/minirocket_xapp_ue.joblib")
fi

# Add offline model if specified
if [ -n "$OFFLINE_MODEL" ] && [ -f "$OFFLINE_MODEL" ]; then
    ARGS+=("--offline-model" "$OFFLINE_MODEL")
fi

# Add use-llm flag (Now controlled by the variable above)
if [ "$USE_LLM" = "true" ]; then
    ARGS+=("--use-llm")
fi

# Add commands-disabled flag if specified
if [ "$COMMANDS_DISABLED" = "true" ]; then
    ARGS+=("--commands-disabled")
fi

# Add model type for loss history filename
MODEL_TYPE="${MODEL_TYPE:-stable}"
ARGS+=("--model-type" "$MODEL_TYPE")

# Add loss history file if specified
if [ -n "$LOSS_HISTORY_FILE" ]; then
    ARGS+=("--loss-history-file" "$LOSS_HISTORY_FILE")
fi

# Add SLO file if specified
SLO_FILE="${SLO_FILE:-configs/slos.json}"
ARGS+=("--slo-file" "$SLO_FILE")

echo "=========================================="
echo "Starting AI System on AMD GPU (ROCm 6.1.2)"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Using LLM: $USE_LLM"
echo "Log Level: $LOG_LEVEL"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "=========================================="

# Function to plot loss after training
plot_loss_function() {
    if [ "$PLOT_LOSS" = "true" ]; then
        echo "=========================================="
        echo "Plotting loss function..."
        echo "=========================================="
        
        if [ -f "$STABLE_LOSS_FILE" ] || [ -f "$UNSTABLE_LOSS_FILE" ]; then
            PLOT_ARGS=()
            [ -f "$STABLE_LOSS_FILE" ] && PLOT_ARGS+=("--stable" "$STABLE_LOSS_FILE")
            [ -f "$UNSTABLE_LOSS_FILE" ] && PLOT_ARGS+=("--unstable" "$UNSTABLE_LOSS_FILE")
            PLOT_ARGS+=("--output" "$LOSS_PLOT_OUTPUT")
            
            # Using ROCm python for plotting too
            /opt/rocm_sdk_612/bin/python3 plot_loss.py "${PLOT_ARGS[@]}"
            echo "Loss plot saved to: $LOSS_PLOT_OUTPUT"
        fi
    fi
}

# Set up trap to plot loss on exit
if [ "$PLOT_LOSS" = "true" ]; then
    trap plot_loss_function EXIT
fi

# RUN THE AI
# Note: we use the absolute path to the ROCm python binary to ensure it ignores system python
/opt/rocm_sdk_612/bin/python3 "$PROJECT_ROOT/src/demo/ain/RL_demo/xapp_demo_membus.py" "${ARGS[@]}"

# Plot loss after training completes (if not already done by trap)
if [ "$PLOT_LOSS" = "true" ]; then
    plot_loss_function
fi