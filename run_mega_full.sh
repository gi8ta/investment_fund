#!/bin/bash
# Standalone runner for the "mega_full" benchmark — every subsystem ON,
# 3-round investment debates, 3-round risk debates. Designed to be
# launched by a watcher AFTER the main run_ablation.sh suite finishes.

cd "$(dirname "$0")"
mkdir -p results

# Skip if already complete (idempotent).
if [[ -f results/mega_full.json ]] && grep -q '"status": "complete"' results/mega_full.json 2>/dev/null; then
    echo "[$(date)] mega_full already complete — skipping"
    exit 0
fi

# Drop SQLite api_cache before run to avoid concurrent-write contention.
rm -f fund/dataflows/data_cache/api_cache.db 2>/dev/null

echo "============================================"
echo "[$(date)] START: mega_full"
echo "  every subsystem ON, debate-rounds=3, risk-rounds=3"
echo "============================================"

python backtest.py \
    --tickers NVDA AAPL MSFT TSLA SPY \
    --start 2025-01-06 --end 2025-06-30 \
    --freq-weeks 1 --hold-days 7 \
    --max-workers 3 \
    --debate-rounds 3 --risk-rounds 3 \
    --provider openrouter \
    --deep-model openai/gpt-4o-mini \
    --quick-model openai/gpt-4o-mini \
    --backend-url https://openrouter.ai/api/v1 \
    --enable-darwinian \
    --enable-cro \
    --enable-forward-context \
    --enable-autoresearch \
    --enable-janus \
    --output results/mega_full.json \
    --verbose-log results/mega_full.log

rc=$?
echo "[$(date)] mega_full exit=$rc"

# Re-run summary so the final SUMMARY.md includes mega_full.
python summarize.py results/ > results/SUMMARY.md 2>&1 || true
echo "[$(date)] summary regenerated"
