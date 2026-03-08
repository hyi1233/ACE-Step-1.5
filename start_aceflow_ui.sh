#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Minimal ACEFlow UI launcher for standard ACE-Step uv workflow
# Optional overrides before launch:
#   export PORT=7861
#   export SERVER_NAME=127.0.0.1
#   export ACEFLOW_CONFIG_PATH=acestep-v15-turbo
#   export ACEFLOW_LM_MODEL_PATH=acestep-5Hz-lm-4B
#   export ACEFLOW_DEVICE=auto
#   export ACEFLOW_RESULTS_DIR="$SCRIPT_DIR/aceflow_outputs"

: "${PORT:=7861}"
: "${SERVER_NAME:=127.0.0.1}"
: "${ACEFLOW_CONFIG_PATH:=acestep-v15-turbo}"
: "${ACEFLOW_LM_MODEL_PATH:=acestep-5Hz-lm-4B}"
: "${ACEFLOW_DEVICE:=auto}"
: "${ACEFLOW_RESULTS_DIR:=$SCRIPT_DIR/aceflow_outputs}"

if ! command -v uv >/dev/null 2>&1; then
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "[ACEFLOW] uv not found in PATH."
        exit 1
    fi
fi

if [[ ! -x ".venv/bin/python" ]]; then
    echo "[ACEFLOW] .venv not found, running uv sync..."
    uv sync
fi

export ACESTEP_REMOTE_CONFIG_PATH="$ACEFLOW_CONFIG_PATH"
export ACESTEP_REMOTE_LM_MODEL_PATH="$ACEFLOW_LM_MODEL_PATH"
export ACESTEP_REMOTE_DEVICE="$ACEFLOW_DEVICE"
export ACESTEP_REMOTE_RESULTS_DIR="$ACEFLOW_RESULTS_DIR"

echo "Starting ACEFlow UI..."
echo "http://$SERVER_NAME:$PORT"
echo "[ACEFLOW] CFG=$ACESTEP_REMOTE_CONFIG_PATH | LM=$ACESTEP_REMOTE_LM_MODEL_PATH"

source .venv/bin/activate
python -m acestep.ui.aceflow.run --host "$SERVER_NAME" --port "$PORT"
