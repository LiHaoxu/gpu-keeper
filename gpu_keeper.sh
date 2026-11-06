#!/usr/bin/env bash
# gpu_keeper.sh — keep at least $TARGET GPUs reserved for your own experiments on a shared machine.
#
# Every $POLL seconds the keeper
#   1. restarts any holder container of an already-held GPU that is not running;
#   2. if fewer than $TARGET GPUs are held, claims GPUs whose used memory is below $IDLE_MIB MiB by starting a
#      memory-only holder container on each of them.
# A holder allocates (free memory - $RESERVE_GB GiB) on its GPU and then sleeps: the GPU looks busy to others while
# $RESERVE_GB GiB stay available for your own runs. Held GPU ids are listed in $STATE_DIR/held.
# Undo everything with gpu_release.sh.
#
# usage:  TARGET=4 RESERVE_GB=24 setsid nohup ./gpu_keeper.sh >> keeper.log 2>&1 < /dev/null &
set -uo pipefail

STATE_DIR=${STATE_DIR:-$HOME/.gpu_keeper}      # held list, keeper pid
TARGET=${TARGET:-2}                            # how many GPUs to hold
IDLE_MIB=${IDLE_MIB:-100}                      # a GPU counts as idle below this many MiB used
RESERVE_GB=${RESERVE_GB:-16}                   # memory each holder leaves free for your own jobs
IMAGE=${IMAGE:-nvcr.io/nvidia/pytorch:25.06-py3}   # any image with python + torch + CUDA
PREFIX=${PREFIX:-gpu-hold-g}                   # holder container name = $PREFIX<gpu id>
CANDIDATES=${CANDIDATES:-}                     # e.g. "0,1,2,3"; empty = any GPU
POLL=${POLL:-30}                               # seconds between checks

mkdir -p "$STATE_DIR"
HELD=$STATE_DIR/held
touch "$HELD"
echo $$ > "$STATE_DIR/keeper.pid"

log() { echo "$(date '+%F %T') $*"; }

start_holder() {  # <gpu>
    docker rm "$PREFIX$1" >/dev/null 2>&1       # a stopped holder of the same name, if any
    docker run -d --name "$PREFIX$1" --restart unless-stopped --gpus "\"device=$1\"" -u "$(id -u):$(id -g)" \
        "$IMAGE" python -c "
import time, torch
free, total = torch.cuda.mem_get_info()
n = max(0, free - int($RESERVE_GB * 2**30))
hold = torch.empty(n, dtype=torch.uint8, device='cuda')
print(f'holding {n / 2**30:.1f} GiB of {total / 2**30:.1f} GiB, leaving $RESERVE_GB GiB free', flush=True)
while True:
    time.sleep(3600)
" >/dev/null
}

idle_gpus() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v m="$IDLE_MIB" '$2 < m {print $1}'
}

is_candidate() {
    [ -z "$CANDIDATES" ] && return 0
    case ",$CANDIDATES," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

stop() {
    log "keeper stopped (holders keep running; use gpu_release.sh to free the GPUs)"
    rm -f "$STATE_DIR/keeper.pid"
    exit 0
}
trap stop TERM INT

log "keeper started (pid $$): target $TARGET GPUs, idle < $IDLE_MIB MiB, reserve $RESERVE_GB GiB each"
while true; do
    for g in $(cat "$HELD"); do
        state=$(docker inspect -f '{{.State.Running}}' "$PREFIX$g" 2>/dev/null || echo missing)
        if [ "$state" != true ]; then
            log "holder on GPU $g is $state, restarting"
            start_holder "$g"
        fi
    done
    if [ "$(grep -c . "$HELD")" -lt "$TARGET" ]; then
        for g in $(idle_gpus); do
            grep -qx "$g" "$HELD" && continue
            is_candidate "$g" || continue
            if start_holder "$g"; then
                echo "$g" >> "$HELD"
                log "claimed GPU $g"
            fi
            [ "$(grep -c . "$HELD")" -ge "$TARGET" ] && break
        done
    fi
    sleep "$POLL" &
    wait $!
done
