#!/bin/bash
# Watcher: waits for the main run_ablation.sh suite (PID passed as $1)
# to exit, then launches run_mega_full.sh.
# Detached from the launching shell — survives terminal close.

cd "$(dirname "$0")"

WAIT_PID="${1:?usage: run_mega_after.sh <ablation_pid>}"

echo "[$(date)] watcher started, waiting for PID $WAIT_PID to exit ..." \
    >> results/mega_watcher.log

# Poll because `wait` only works on direct children of this shell.
while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $WAIT_PID exited — launching mega_full" \
    >> results/mega_watcher.log

bash run_mega_full.sh >> results/mega_full_console.log 2>&1

echo "[$(date)] mega_full finished" >> results/mega_watcher.log
