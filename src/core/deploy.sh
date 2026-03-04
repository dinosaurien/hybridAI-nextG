#!/bin/bash
# Deploy script for AI Core System (Telemetry, Orchestrator, UI)
# Usage: ./deploy_core.sh

set -e

export PROJECT_ROOT="/home/exposed/Desktop/hybridAI-nextG"
export PYTHONPATH="$PROJECT_ROOT/src/demo:$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6000}"
WEB_PORT="${WEB_PORT:-8080}"
TARGET_METRIC="${TARGET_METRIC:-DRB_PdcpSduDelayDl}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed. Please install it or check your PATH."
    exit 1
fi

# Ensure Poetry is using the local virtual environment based on the lock file
echo "Verifying Poetry environment..."
poetry install --no-root # Installs dependencies from poetry.lock quickly if missing

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

echo "=========================================="
echo "Starting Core AI System via Poetry"
echo "=========================================="
echo "xApp TCP Host : $HOST"
echo "xApp TCP Port : $PORT"
echo "Web UI Port   : $WEB_PORT"
echo "Target Metric : $TARGET_METRIC"
echo "Log Level     : $LOG_LEVEL"
echo "=========================================="

# RUN THE CORE SYSTEM USING POETRY
poetry run python src/core/main.py "${ARGS[@]}"