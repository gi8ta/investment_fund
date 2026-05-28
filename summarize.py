"""Summarize all backtest result JSONs in a directory into one markdown table.

Usage: python summarize.py results/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev


def load_runs(results_dir: Path):
    runs = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            print(f"<!-- {f.name}: read error {exc} -->")
            continue
        runs.append((f.stem, data))
    return runs


def trade_stats(trades):
    n = len(trades)
    if n == 0:
        return None
    rets = [t.get("return_pct", 0.0) for t in trades]
    wins = [t for t in trades if t.get("profitable")]
    signal_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for t in trades:
        s = t.get("signal", "")
        if s in signal_counts:
            signal_counts[s] += 1
    avg = mean(rets)
    sd = stdev(rets) if n > 1 else 0.0
    sharpe_like = (avg / sd) if sd > 0 else 0.0
    return {
        "n": n,
        "avg_return_pct": avg,
        "stdev_return_pct": sd,
        "sharpe_like": sharpe_like,
        "hit_rate": len(wins) / n,
        "total_pnl_pct": sum(rets),
        "buy": signal_counts["BUY"],
        "sell": signal_counts["SELL"],
        "hold": signal_counts["HOLD"],
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize.py <results_dir>")
        sys.exit(2)

    results_dir = Path(sys.argv[1])
    runs = load_runs(results_dir)

    print("# Benchmark Summary")
    print()
    print(f"_Generated from `{results_dir}` — {len(runs)} runs._")
    print()

    print("| Run | Status | N trades | Avg ret % | Sharpe-like | Hit rate | Total PnL % | B/S/H |")
    print("|---|---|---:|---:|---:|---:|---:|---|")

    for name, data in runs:
        status = data.get("status", "unknown")
        trades = data.get("trades", [])
        stats = trade_stats(trades)
        if stats is None:
            print(f"| {name} | {status} | 0 | — | — | — | — | — |")
            continue
        print(
            f"| {name} | {status} | {stats['n']} | "
            f"{stats['avg_return_pct']:+.2f} | {stats['sharpe_like']:+.2f} | "
            f"{stats['hit_rate']:.1%} | {stats['total_pnl_pct']:+.2f} | "
            f"{stats['buy']}/{stats['sell']}/{stats['hold']} |"
        )

    print()
    print("## Errors")
    print()
    any_err = False
    for name, data in runs:
        errs = data.get("errors") or []
        if errs:
            any_err = True
            print(f"### {name} ({len(errs)} errors)")
            for e in errs[:5]:
                print(f"- {e}")
            if len(errs) > 5:
                print(f"- … and {len(errs) - 5} more")
            print()
    if not any_err:
        print("None.")


if __name__ == "__main__":
    main()
