#!/bin/bash
#
# Open a bigpurple Jupyter session in one command.
# Place this somewhere in your $PATH (e.g. ~/bin/bp_connect.sh) and run it
# on your laptop.
#
# What it does:
#   1) ssh to bigpurple
#   2) ./start_jupyter.sh — reuse or submit the GPU+Jupyter job, wait until ready
#   3) Open an SSH tunnel from laptop -> bigpurple-login -> compute_node:port
#   4) Print the URL with token. Browse there.
#
# Tunnel runs in the foreground. Ctrl-C when done; the bigpurple job keeps
# running (4-hour walltime) so a re-run of this script just re-attaches.
#
# Requires:
#   - your ssh access to bigpurple works (`ssh bigpurple echo ok` or full FQDN)
#   - start_jupyter.sh exists at /gpfs/data/oermannlab/users/schula12/Surgical-SAM-2/
#     (it does — pulled in by the git push)

set -euo pipefail

BP_HOST="${BP_HOST:-bigpurple}"     # override with `BP_HOST=bigpurple.nyumc.org bp_connect.sh`
REPO="${BP_REPO:-/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2}"
LOCAL_PORT="${LOCAL_PORT:-9001}"

echo "==> Asking $BP_HOST for a Jupyter session ..."
VARS=$(ssh -o ConnectTimeout=10 "$BP_HOST" "cd $REPO && ./start_jupyter.sh")
eval "$VARS"

if [ -z "${NODE:-}" ] || [ -z "${PORT:-}" ] || [ -z "${TOKEN:-}" ]; then
    echo "Failed to read NODE/PORT/TOKEN from start_jupyter.sh output:"
    echo "$VARS"
    exit 1
fi

URL="http://127.0.0.1:$LOCAL_PORT/lab?token=$TOKEN"
echo
echo "Jupyter is alive on $BP_HOST:$NODE port $PORT (job $JOB)."
echo
echo "Opening tunnel  laptop:$LOCAL_PORT  ->  $NODE:$PORT"
echo "Browse to:"
echo "  $URL"
echo
# Best-effort: try to open the browser automatically (macOS, Linux).
( command -v open    >/dev/null && open    "$URL" ) >/dev/null 2>&1 || \
( command -v xdg-open >/dev/null && xdg-open "$URL" ) >/dev/null 2>&1 || true

# Hold the tunnel open until the user Ctrl-Cs.
exec ssh -N -L "$LOCAL_PORT:$NODE:$PORT" "$BP_HOST"
