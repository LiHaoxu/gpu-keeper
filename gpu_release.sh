#!/usr/bin/env bash
# gpu_release.sh — give the GPUs back: stop the keeper (by its pid file), stop and remove every holder container,
# and empty the held list. Running experiments are not touched.
# usage:  ./gpu_release.sh            (same STATE_DIR / PREFIX as the keeper)
set -uo pipefail

STATE_DIR=${STATE_DIR:-$HOME/.gpu_keeper}
PREFIX=${PREFIX:-gpu-hold-g}

if [ -f "$STATE_DIR/keeper.pid" ]; then
    pid=$(cat "$STATE_DIR/keeper.pid")
    if kill "$pid" 2>/dev/null; then
        echo "stopped keeper (pid $pid)"
    else
        echo "keeper pid $pid was not running"
    fi
    rm -f "$STATE_DIR/keeper.pid"
fi

ids=$(docker ps -aq --filter "name=^${PREFIX}")
if [ -n "$ids" ]; then
    docker stop $ids >/dev/null
    docker rm $ids >/dev/null
    echo "removed $(echo "$ids" | wc -l | tr -d ' ') holder container(s)"
fi
: > "$STATE_DIR/held" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
