#!/bin/bash
# Watcher: as soon as no_risk_debate completes, plant a stub
# with_janus.json marked status=complete so the running ablation script
# (which still has with_janus in its in-memory queue) skips it.
cd "$(dirname "$0")"

while true; do
    if [[ -f results/no_risk_debate.json ]] && \
       grep -q '"status": "complete"' results/no_risk_debate.json 2>/dev/null; then
        if [[ ! -f results/with_janus.json ]] || \
           ! grep -q '"status": "complete"' results/with_janus.json 2>/dev/null; then
            cat > results/with_janus.json <<EOF
{
  "status": "complete",
  "as_of": "$(date -u +%Y-%m-%dT%H:%M:%S)",
  "skipped_reason": "manually skipped to free time for mega_full",
  "trades": [],
  "errors": []
}
EOF
            echo "[$(date)] planted stub with_janus.json — ablation will skip it" \
                >> results/skip_janus.log
        fi
        break
    fi
    sleep 60
done
