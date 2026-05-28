"""Пересчёт результатов бэктеста с гипотетическим стоп-лоссом.

Использует кэшированные дневные OHLC из fund/dataflows/data_cache/*.csv,
чтобы для каждой сделки проверить, не пробивал ли внутридневной диапазон
порог стоп-лосса в течение периода удержания.

Учитывается gap-исполнение: если день открылся за стопом, выход
происходит по open, а не по уровню стопа. Это даёт реалистичный
slippage в части убыточных сделок.

Использование:
    python stop_loss_analysis.py <results_dir> [--stop 0.03]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def find_cache_file(ticker: str, cache_dir: Path) -> Optional[Path]:
    candidates = sorted(cache_dir.glob(f"{ticker}-YFin-data-*.csv"), reverse=True)
    return candidates[0] if candidates else None


def load_ohlc(ticker: str, cache_dir: Path) -> Optional[pd.DataFrame]:
    f = find_cache_file(ticker, cache_dir)
    if f is None:
        return None
    df = pd.read_csv(f)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df.set_index("Date")[["Open", "High", "Low", "Close"]]


def apply_stop(trade: dict, ohlc: pd.DataFrame, stop_pct: float) -> dict:
    """Возвращает копию сделки с пересчитанным exit_price если стоп пробит."""
    out = dict(trade)
    sig = trade.get("signal")
    if sig not in ("BUY", "SELL"):
        return out

    entry = trade["entry_price"]
    entry_date = trade["signal_date"]
    exit_date = trade["exit_date"]

    days_between = ohlc.loc[
        (ohlc.index > entry_date) & (ohlc.index <= exit_date)
    ]
    if days_between.empty:
        return out

    if sig == "BUY":
        stop_level = entry * (1 - stop_pct)
        for date, row in days_between.iterrows():
            if row["Low"] <= stop_level:
                fill = row["Open"] if row["Open"] <= stop_level else stop_level
                ret = (fill - entry) / entry * 100
                out["exit_price"] = round(fill, 4)
                out["exit_date"] = date
                out["return_pct"] = round(ret, 4)
                out["profitable"] = ret > 0
                out["stop_loss_fired"] = True
                out["gap_fill"] = row["Open"] <= stop_level
                return out
    else:  # SELL
        stop_level = entry * (1 + stop_pct)
        for date, row in days_between.iterrows():
            if row["High"] >= stop_level:
                fill = row["Open"] if row["Open"] >= stop_level else stop_level
                ret = -((fill - entry) / entry) * 100
                out["exit_price"] = round(fill, 4)
                out["exit_date"] = date
                out["return_pct"] = round(ret, 4)
                out["profitable"] = ret > 0
                out["stop_loss_fired"] = True
                out["gap_fill"] = row["Open"] >= stop_level
                return out

    out["stop_loss_fired"] = False
    return out


def analyse_run(path: Path, ohlc_cache: dict, stop_pct: float) -> Optional[dict]:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    trades = data.get("trades") or []
    if not trades:
        return None

    new_trades = []
    fired = 0
    gap = 0
    for t in trades:
        ticker = t["ticker"]
        if ticker not in ohlc_cache:
            return None
        nt = apply_stop(t, ohlc_cache[ticker], stop_pct)
        new_trades.append(nt)
        if nt.get("stop_loss_fired"):
            fired += 1
            if nt.get("gap_fill"):
                gap += 1

    def equity(trades_list, pct=0.10, capital=10000):
        cap = capital
        peak = cap
        max_dd = 0.0
        for t in sorted(trades_list, key=lambda x: x["signal_date"]):
            if t["signal"] in ("BUY", "SELL"):
                cap += cap * pct * t["return_pct"] / 100
            peak = max(peak, cap)
            max_dd = max(max_dd, (peak - cap) / peak)
        return cap, max_dd

    cap_o, dd_o = equity(trades)
    cap_n, dd_n = equity(new_trades)

    active_o = [t for t in trades if t["signal"] in ("BUY", "SELL")]
    active_n = [t for t in new_trades if t["signal"] in ("BUY", "SELL")]
    hit_o = sum(1 for t in active_o if t["profitable"]) / len(active_o) * 100 if active_o else 0
    hit_n = sum(1 for t in active_n if t["profitable"]) / len(active_n) * 100 if active_n else 0

    augmented = dict(data)
    augmented["trades"] = new_trades
    augmented["stop_loss_pct"] = stop_pct
    out_path = path.with_name(path.stem + "_with_stoploss" + path.suffix)
    out_path.write_text(json.dumps(augmented, indent=2, ensure_ascii=False))

    return {
        "name": path.stem,
        "n": len(trades),
        "fired": fired,
        "gap": gap,
        "roi_orig": (cap_o - 10000) / 100,
        "roi_stop": (cap_n - 10000) / 100,
        "dd_orig": dd_o * 100,
        "dd_stop": dd_n * 100,
        "hit_orig": hit_o,
        "hit_stop": hit_n,
    }


def main():
    parser = argparse.ArgumentParser(description="Stop-loss what-if analysis on backtest results")
    parser.add_argument("results_dir", help="Directory with *.json result files")
    parser.add_argument("--stop", type=float, default=0.03, help="Stop-loss level, default 0.03 = 3%")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cache_dir = Path(__file__).parent / "fund" / "dataflows" / "data_cache"

    tickers = ["NVDA", "AAPL", "MSFT", "TSLA", "SPY"]
    ohlc_cache = {}
    for t in tickers:
        df = load_ohlc(t, cache_dir)
        if df is None:
            print(f"WARN: no OHLC cache for {t}", file=sys.stderr)
            continue
        ohlc_cache[t] = df

    skip = {"baseline_random", "baseline_momentum", "baseline_buyhold"}
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        if f.stem.endswith("_with_stoploss") or f.stem.endswith("_partial") or f.stem.endswith("_partial2"):
            continue
        if f.stem in skip:
            continue
        r = analyse_run(f, ohlc_cache, args.stop)
        if r:
            rows.append(r)

    if not rows:
        print("no runs analysed")
        return

    print("=" * 110)
    print(f"STOP-LOSS ANALYSIS  (level = -{int(args.stop*100)}%, capital $10000, position 10%)")
    print("=" * 110)
    print(f"{'Run':22s} {'n':>5s} {'fired':>7s} {'gap':>5s}  |  "
          f"{'ROI orig':>9s} {'ROI stop':>9s} {'ΔROI':>7s} |  "
          f"{'DD orig':>8s} {'DD stop':>8s} |  {'hit orig':>9s} {'hit stop':>9s}")
    print("-" * 110)
    for r in rows:
        delta = r["roi_stop"] - r["roi_orig"]
        print(f"{r['name']:22s} {r['n']:>5d} {r['fired']:>7d} {r['gap']:>5d}  |  "
              f"{r['roi_orig']:>+8.2f}% {r['roi_stop']:>+8.2f}% {delta:>+6.2f}  | "
              f"-{r['dd_orig']:>6.2f}% -{r['dd_stop']:>6.2f}% | "
              f"   {r['hit_orig']:>5.1f}%    {r['hit_stop']:>5.1f}%")

    print(f"\nSaved augmented files: {len(rows)} *_with_stoploss.json in {results_dir}/")


if __name__ == "__main__":
    main()
