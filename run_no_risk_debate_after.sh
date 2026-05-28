#!/bin/bash
# Watcher: waits for mega_full python (PID passed as $1) to exit,
# then runs the no_risk_debate ablation with the risk_manager fix.

cd "$(dirname "$0")"
WAIT_PID="${1:?usage: run_no_risk_debate_after.sh <mega_full_pid>}"

echo "[$(date)] watcher started, waiting for PID $WAIT_PID (mega_full) to exit ..." \
    >> results/no_risk_debate_watcher.log

while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done

echo "[$(date)] PID $WAIT_PID exited — launching no_risk_debate" \
    >> results/no_risk_debate_watcher.log

# Skip if a previous run already finished.
if [[ -f results/no_risk_debate.json ]] && \
   grep -q '"status": "complete"' results/no_risk_debate.json 2>/dev/null; then
    echo "[$(date)] no_risk_debate already complete — skipping" \
        >> results/no_risk_debate_watcher.log
    exit 0
fi

rm -f fund/dataflows/data_cache/api_cache.db 2>/dev/null

python backtest.py \
    --tickers NVDA AAPL MSFT TSLA SPY \
    --start 2025-01-06 --end 2025-06-30 \
    --freq-weeks 1 --hold-days 7 \
    --max-workers 3 \
    --debate-rounds 1 --risk-rounds 1 \
    --provider openrouter \
    --deep-model openai/gpt-4o-mini \
    --quick-model openai/gpt-4o-mini \
    --backend-url https://openrouter.ai/api/v1 \
    --enable-darwinian --enable-cro --enable-forward-context \
    --skip-risk-debate \
    --output results/no_risk_debate.json \
    --verbose-log results/no_risk_debate.log \
    >> results/no_risk_debate_console.log 2>&1

echo "[$(date)] no_risk_debate finished" >> results/no_risk_debate_watcher.log

# Regenerate summary
python summarize.py results/ > results/SUMMARY.md 2>&1 || true
