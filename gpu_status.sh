#!/usr/bin/env bash
# gpu_status.sh — is the keeper alive, which GPUs are held, and how much memory is used on every GPU.
set -uo pipefail

STATE_DIR=${STATE_DIR:-$HOME/.gpu_keeper}
PREFIX=${PREFIX:-gpu-hold-g}

if [ -f "$STATE_DIR/keeper.pid" ] && kill -0 "$(cat "$STATE_DIR/keeper.pid")" 2>/dev/null; then
    echo "keeper: running (pid $(cat "$STATE_DIR/keeper.pid"))"
else
    echo "keeper: not running"
fi
echo "held:   $(paste -sd, "$STATE_DIR/held" 2>/dev/null)"
docker ps -a --filter "name=^${PREFIX}" --format '        {{.Names}}  {{.Status}}'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
