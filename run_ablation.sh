#!/bin/bash
# Overnight benchmark: baseline configurations + per-subsystem ablation +
# graph-structure ablation. Each run is isolated — a failure in one run
# does not abort the rest. Results are written incrementally inside each
# run as well, so a hard crash mid-run still leaves a usable JSON.
#
# Estimated time: ~10-12 h on gpt-4o-mini @ 3 parallel workers,
# 5 tickers × 26 weekly signals × 8 runs.
#
# Usage:
#   nohup bash run_ablation.sh > results/console.log 2>&1 &
#   tail -f results/console.log

cd "$(dirname "$0")"
mkdir -p results

# ─── Universe & period ───
TICKERS=(NVDA AAPL MSFT TSLA SPY)
START="2025-01-06"
END="2025-06-30"

COMMON_ARGS=(
    --tickers "${TICKERS[@]}"
    --start "$START" --end "$END"
    --freq-weeks 1 --hold-days 7
    --max-workers 3
    --debate-rounds 1 --risk-rounds 1
    --provider openrouter
    --deep-model openai/gpt-4o-mini
    --quick-model openai/gpt-4o-mini
    --backend-url https://openrouter.ai/api/v1
)

# Three additive subsystems used in the main configuration.
# JANUS and Autoresearch are tested separately, not bundled into the
# baseline_full, because they change the comparison semantics:
#   - JANUS doubles API load via cohorts
#   - Autoresearch mutates prompts mid-run, contaminating A/B
ATLAS_FEATURES=(--enable-darwinian --enable-cro --enable-forward-context)

# ─── Run definitions: name | extra args ───
# Order chosen so the most important results land first.
# mega_full uses 3-round debates and bundles every subsystem incl.
# autoresearch and janus. Its extra args re-pass --debate-rounds and
# --risk-rounds; argparse takes the last value, overriding COMMON_ARGS.
RUNS=(
    "baseline_minimal|"
    "baseline_full|${ATLAS_FEATURES[*]}"
    "no_darwinian|--enable-cro --enable-forward-context"
    "no_cro|--enable-darwinian --enable-forward-context"
    "no_forward_context|--enable-darwinian --enable-cro"
    "no_invest_debate|${ATLAS_FEATURES[*]} --skip-invest-debate"
    "no_risk_debate|${ATLAS_FEATURES[*]} --skip-risk-debate"
    # with_janus skipped — its effect is already captured inside mega_full,
    # and the standalone 5h+ run isn't worth the time given the diploma
    # deadline. Re-enable by uncommenting if needed.
    # "with_janus|${ATLAS_FEATURES[*]} --enable-janus"
    "mega_full|${ATLAS_FEATURES[*]} --enable-autoresearch --enable-janus --debate-rounds 3 --risk-rounds 3"
)

run_one() {
    local name="$1"
    local extra_str="$2"
    local out="results/${name}.json"
    local log="results/${name}.log"

    # If a previous run already finished, skip — useful when restarting
    # after a partial overnight. A partial in_progress file is overwritten.
    if [[ -f "$out" ]] && grep -q '"status": "complete"' "$out" 2>/dev/null; then
        echo "[$(date)] SKIP $name (already complete)"
        return 0
    fi

    echo ""
    echo "============================================"
    echo "[$(date)] START: $name"
    echo "  extra: $extra_str"
    echo "============================================"

    # Clean shared SQLite api_cache before each run. The cache picks up
    # contention under multi-worker concurrent writes and starts emitting
    # "bad parameter or other API misuse" errors that surface as
    # no_entry_price skips. CSV price files are kept (they're append-only).
    rm -f fund/dataflows/data_cache/api_cache.db 2>/dev/null

    # extra_str is intentionally word-split.
    # `set +e` ensures a Python crash here does not kill the script.
    set +e
    python backtest.py "${COMMON_ARGS[@]}" $extra_str \
        --output "$out" \
        --verbose-log "$log" &
    local py_pid=$!

    # Stall watchdog: if $out hasn't been touched for 30 minutes, kill the
    # backtest. The in-process LLM/propagate timeouts catch most hangs;
    # this is a final backstop against any new failure mode.
    (
        local last_mtime=0
        while kill -0 "$py_pid" 2>/dev/null; do
            sleep 60
            if [[ -f "$out" ]]; then
                local cur=$(stat -f %m "$out" 2>/dev/null || echo 0)
                if [[ $cur -gt $last_mtime ]]; then
                    last_mtime=$cur
                    last_seen=$(date +%s)
                fi
            fi
            local now=$(date +%s)
            if [[ -n "${last_seen:-}" ]] && (( now - last_seen > 3600 )); then
                echo "[$(date)] WATCHDOG: $name stalled >60min — killing"
                kill -9 "$py_pid" 2>/dev/null
                break
            fi
        done
    ) &
    local wd_pid=$!

    wait "$py_pid"
    local rc=$?
    kill "$wd_pid" 2>/dev/null
    set -e

    if [[ $rc -eq 0 ]]; then
        echo "[$(date)] DONE  $name"
    else
        echo "[$(date)] FAIL  $name (exit=$rc) — partial results in $out"
    fi
}

echo "============================================"
echo "  OVERNIGHT BENCHMARK SUITE — $(date)"
echo "  Tickers: ${TICKERS[*]}"
echo "  Period:  $START → $END"
echo "  Runs:    ${#RUNS[@]}"
echo "============================================"

start_time=$(date +%s)

for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    extra="${entry#*|}"
    run_one "$name" "$extra"
done

elapsed=$(( $(date +%s) - start_time ))
hours=$(( elapsed / 3600 ))
mins=$(( (elapsed % 3600) / 60 ))

echo ""
echo "============================================"
echo "  ALL RUNS FINISHED — $(date)"
echo "  Total elapsed: ${hours}h ${mins}m"
echo "============================================"

# Generate final summary
python summarize.py results/ > results/SUMMARY.md 2>&1 || \
    echo "[$(date)] WARN: summary generation failed; raw JSONs still in results/"

echo "Summary written to results/SUMMARY.md"
