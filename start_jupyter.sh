#!/bin/bash
#
# Single command to get a working Jupyter session on a bigpurple A100.
# Idempotent: reuses an existing surgsam2_jupyter job if there is one running.
#
# Usage (on bigpurple):
#     ./start_jupyter.sh
#
# Output: shell-eval-friendly NODE=... PORT=... TOKEN=... lines on stdout,
# everything else on stderr. So a caller can do:
#     eval "$(ssh bigpurple ./start_jupyter.sh)"
# and then open the tunnel.

set -euo pipefail
REPO=/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
cd "$REPO"
mkdir -p logs

log() { echo "$@" >&2; }

# 1) Is there already a running surgsam2_jupyter job?
EXISTING_JOB=$(squeue -u "$USER" -h -o '%i %j %T' 2>/dev/null \
    | awk '$2=="surgsam2_jupyter" && $3=="RUNNING" { print $1; exit }')

if [ -n "$EXISTING_JOB" ]; then
    JOB="$EXISTING_JOB"
    log "Reusing existing job $JOB"
else
    log "No running surgsam2_jupyter job; submitting a new one ..."
    SUB=$(sbatch run_jupyter.sh | tail -1)
    JOB=$(echo "$SUB" | awk '{print $NF}')
    log "Submitted job $JOB; waiting for RUNNING state ..."

    # Wait up to ~10 min for the job to start (queue can be slow).
    for _ in $(seq 1 120); do
        STATE=$(squeue -j "$JOB" -h -o '%T' 2>/dev/null || echo UNKNOWN)
        if [ "$STATE" = "RUNNING" ]; then break; fi
        sleep 5
    done
    if [ "$STATE" != "RUNNING" ]; then
        log "Job $JOB not RUNNING after 10 min (state=$STATE). Inspect 'squeue -j $JOB'."
        exit 1
    fi
fi

LOG="logs/jupyter_${JOB}.out"
ERR="logs/jupyter_${JOB}.err"

# 2) Wait until the Jupyter URL line appears in the log (handles bootstrap).
log "Waiting for Jupyter URL to appear in $ERR ..."
TOKEN=""
for _ in $(seq 1 240); do
    TOKEN=$(grep -m1 -oE 'http://127\.0\.0\.1:[0-9]+/lab\?token=[a-f0-9]+' "$ERR" 2>/dev/null \
            | head -1 | sed -E 's|.*token=||')
    if [ -n "$TOKEN" ]; then break; fi
    sleep 2
done
if [ -z "$TOKEN" ]; then
    log "Could not find token in $ERR after 8 min."
    log "Tail of $ERR:"
    tail -20 "$ERR" >&2 || true
    exit 1
fi

NODE=$(scontrol show job "$JOB" | sed -nE 's/.*NodeList=([A-Za-z0-9_-]+).*/\1/p' | head -1)
PORT=$(grep -m1 -oE 'http://127\.0\.0\.1:[0-9]+/' "$ERR" \
       | head -1 | sed -E 's|.*:([0-9]+)/.*|\1|')

log ""
log "Ready:  job=$JOB  node=$NODE  port=$PORT"
log ""
log "From your laptop, open the tunnel:"
log "  ssh -N -L 9001:$NODE:$PORT schula12@bigpurple.nyumc.org"
log "Then browse to:"
log "  http://127.0.0.1:9001/lab?token=$TOKEN"
log ""

# Stdout: shell-eval-friendly so the laptop-side wrapper can consume.
echo "JOB=$JOB"
echo "NODE=$NODE"
echo "PORT=$PORT"
echo "TOKEN=$TOKEN"
