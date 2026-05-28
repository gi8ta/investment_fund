#!/usr/bin/env python3
"""
No-LLM baselines for ablation study comparison.

Generates BUY/SELL/HOLD signals using simple rules (no LLM calls),
then calculates the same metrics as the main backtest for fair comparison.

Usage:
    python baselines.py --strategy random --output results/baseline_random.json
    python baselines.py --strategy momentum --output results/baseline_momentum.json
    python baselines.py --strategy buyhold --output results/baseline_buyhold.json
"""

import json
import math
import random
import argparse
from collections import defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd
import yfinance as yf


# ─── Data structures ───────────────────────────────────────────

@dataclass
class TradeResult:
    ticker: str
    signal_date: str
    signal: str
    entry_price: float
    exit_date: str
    exit_price: float
    return_pct: float
    profitable: bool


# ─── Price helpers (same as backtest.py) ───────────────────────

def prefetch_prices(tickers, start_date, end_date, buffer_days=30):
    end_ext = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    # Also extend start for momentum lookback
    start_ext = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
    cache = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_ext, end=end_ext, progress=False, auto_adjust=True)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cache[ticker] = df[["Open", "Close"]].copy()
        except Exception:
            pass
    return cache


def price_on_or_after(ticker, date, cache, col="Close"):
    if ticker not in cache:
        return None
    df = cache[ticker]
    available = df.index[df.index >= pd.Timestamp(date)]
    if available.empty:
        return None
    return float(df.loc[available[0], col])


def next_trading_day(date, n_calendar_days, ticker, cache):
    target = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=n_calendar_days)
    if ticker not in cache:
        return target.strftime("%Y-%m-%d")
    df = cache[ticker]
    available = df.index[df.index >= pd.Timestamp(target)]
    if available.empty:
        return target.strftime("%Y-%m-%d")
    return str(available[0].date())


def signal_dates(start, end, freq_weeks=1):
    cur = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    while cur.weekday() != 0:
        cur += timedelta(days=1)
    dates = []
    while cur <= stop:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(weeks=freq_weeks)
    return dates


# ─── Strategies ────────────────────────────────────────────────

def random_signal(ticker, date, cache, seed_base=42):
    """Uniform random BUY/SELL/HOLD."""
    # Deterministic seed per (ticker, date) for reproducibility
    seed = hash(f"{ticker}_{date}_{seed_base}") % (2**31)
    rng = random.Random(seed)
    return rng.choice(["BUY", "SELL", "HOLD"])


def momentum_signal(ticker, date, cache, lookback_days=20):
    """Simple momentum: if price rose over lookback → BUY, fell → SELL, else HOLD."""
    if ticker not in cache:
        return "HOLD"
    df = cache[ticker]
    target = pd.Timestamp(date)
    lookback_start = target - timedelta(days=lookback_days + 10)  # buffer for trading days
    window = df[(df.index >= lookback_start) & (df.index < target)]
    if len(window) < 5:
        return "HOLD"
    start_price = float(window.iloc[0]["Close"])
    end_price = float(window.iloc[-1]["Close"])
    ret = (end_price - start_price) / start_price
    if ret > 0.02:   # > 2% rise → BUY
        return "BUY"
    elif ret < -0.02:  # > 2% drop → SELL
        return "SELL"
    else:
        return "HOLD"


def buyhold_signal(ticker, date, cache):
    """Always BUY — equivalent to buy-and-hold benchmark."""
    return "BUY"


STRATEGIES = {
    "random": random_signal,
    "momentum": momentum_signal,
    "buyhold": buyhold_signal,
}


# ─── Metrics (same formulas as backtest.py) ────────────────────

def calculate_metrics(results, tickers, cache, start, end, freq_weeks=1):
    active = [r for r in results if r.signal != "HOLD"]
    buy = [r for r in results if r.signal == "BUY"]
    sell = [r for r in results if r.signal == "SELL"]
    hold = [r for r in results if r.signal == "HOLD"]

    n_tickers = max(len(tickers), 1)
    trade_rets = [r.return_pct for r in active]
    win_rate = (sum(1 for r in active if r.profitable) / len(active) * 100) if active else 0.0
    avg_ret = sum(trade_rets) / len(trade_rets) if trade_rets else 0.0

    # Compound portfolio
    capital = 10_000.0
    equity_peak = capital
    max_dd_pct = 0.0
    for r in sorted(results, key=lambda x: (x.signal_date, x.ticker)):
        if r.signal != "HOLD":
            position = capital / n_tickers
            capital += position * (r.return_pct / 100)
            capital = max(capital, 0.0)
            if capital > equity_peak:
                equity_peak = capital
            dd = (equity_peak - capital) / equity_peak * 100
            max_dd_pct = max(max_dd_pct, dd)
    total_ret_compound = (capital - 10_000) / 10_000 * 100

    # Sharpe ratio
    period_buckets = defaultdict(list)
    for r in results:
        period_buckets[r.signal_date].append(r.return_pct if r.signal != "HOLD" else 0.0)
    period_rets = [
        sum(v + [0.0] * (n_tickers - len(v))) / n_tickers
        for v in [period_buckets[d] for d in sorted(period_buckets)]
    ]
    sharpe = 0.0
    if len(period_rets) >= 2:
        mean_p = sum(period_rets) / len(period_rets)
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in period_rets) / (len(period_rets) - 1))
        if std_p > 0:
            ann_factor = math.sqrt(52.0 / freq_weeks)
            sharpe = round((mean_p / std_p) * ann_factor, 3)

    # Sortino ratio
    sortino = 0.0
    if len(period_rets) >= 2:
        mean_p = sum(period_rets) / len(period_rets)
        downside = [min(p, 0) for p in period_rets]
        downside_std = math.sqrt(sum(d ** 2 for d in downside) / (len(downside) - 1))
        if downside_std > 0:
            ann_factor = math.sqrt(52.0 / freq_weeks)
            sortino = round((mean_p / downside_std) * ann_factor, 3)

    calmar = round(total_ret_compound / max_dd_pct, 3) if max_dd_pct > 0 else None

    # Buy-and-hold benchmark
    bh_rets = []
    for t in tickers:
        ep = price_on_or_after(t, start, cache)
        xp = price_on_or_after(t, end, cache)
        if ep and xp:
            bh_rets.append((xp - ep) / ep * 100)
    bh_ret = sum(bh_rets) / len(bh_rets) if bh_rets else 0.0

    return {
        "total_signals": len(results),
        "active_signals": len(active),
        "signal_distribution": {"BUY": len(buy), "SELL": len(sell), "HOLD": len(hold)},
        "win_rate_pct": round(win_rate, 2),
        "avg_return_per_trade_pct": round(avg_ret, 4),
        "best_trade_pct": round(max(trade_rets), 4) if trade_rets else 0,
        "worst_trade_pct": round(min(trade_rets), 4) if trade_rets else 0,
        "portfolio_start": 10_000,
        "portfolio_end_compound": round(capital, 2),
        "total_return_compound_pct": round(total_ret_compound, 4),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "benchmark_buyhold_pct": round(bh_ret, 4),
        "alpha_vs_buyhold_pct": round(total_ret_compound - bh_ret, 4),
    }


# ─── Main loop ─────────────────────────────────────────────────

def run_baseline(
    strategy_name: str,
    tickers: list,
    start_date: str,
    end_date: str,
    hold_days: int = 7,
    freq_weeks: int = 1,
    output_file: str = "baseline_results.json",
):
    strategy_fn = STRATEGIES[strategy_name]

    print(f"\n{'='*60}")
    print(f"  Baseline: {strategy_name.upper()}")
    print(f"  Tickers:  {', '.join(tickers)}")
    print(f"  Period:   {start_date} → {end_date}")
    print(f"  Hold:     {hold_days} days")
    print(f"{'='*60}")

    print("\nFetching prices ...")
    cache = prefetch_prices(tickers, start_date, end_date)
    dates = signal_dates(start_date, end_date, freq_weeks)

    print(f"Signal dates: {len(dates)}, tickers: {len(tickers)}, "
          f"total: {len(dates) * len(tickers)}")

    results = []
    for date in dates:
        for ticker in tickers:
            entry_price = price_on_or_after(ticker, date, cache)
            if entry_price is None:
                continue

            signal = strategy_fn(ticker, date, cache)

            exit_date = next_trading_day(date, hold_days, ticker, cache)
            exit_price = price_on_or_after(ticker, exit_date, cache)
            if exit_price is None:
                continue

            raw_ret = (exit_price - entry_price) / entry_price
            return_pct = (raw_ret * 100 if signal == "BUY"
                          else -raw_ret * 100 if signal == "SELL"
                          else 0.0)

            results.append(TradeResult(
                ticker=ticker,
                signal_date=date,
                signal=signal,
                entry_price=round(entry_price, 4),
                exit_date=exit_date,
                exit_price=round(exit_price, 4),
                return_pct=round(return_pct, 4),
                profitable=return_pct > 0,
            ))

    metrics = calculate_metrics(results, tickers, cache, start_date, end_date, freq_weeks)

    # Monthly breakdown
    monthly_buckets = defaultdict(list)
    for r in results:
        monthly_buckets[r.signal_date[:7]].append(r)
    monthly = {}
    for month in sorted(monthly_buckets.keys()):
        rs = monthly_buckets[month]
        active = [r for r in rs if r.signal != "HOLD"]
        wins = sum(1 for r in active if r.profitable)
        rets = [r.return_pct for r in active]
        monthly[month] = {
            "signals": len(rs),
            "active": len(active),
            "win_rate_pct": round(wins / len(active) * 100, 1) if active else 0.0,
            "avg_return_pct": round(sum(rets) / len(rets), 3) if rets else 0.0,
        }

    payload = {
        "status": "complete",
        "strategy": strategy_name,
        "config": {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
            "hold_days": hold_days,
            "freq_weeks": freq_weeks,
        },
        "trades": [asdict(r) for r in results],
        "metrics": metrics,
        "monthly_breakdown": monthly,
    }

    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    # Print summary
    print(f"\n{'─'*60}")
    print(f"  Strategy:       {strategy_name.upper()}")
    print(f"  Total signals:  {metrics['total_signals']}")
    print(f"  Active:         {metrics['active_signals']}")
    dist = metrics['signal_distribution']
    print(f"  BUY/SELL/HOLD:  {dist['BUY']}/{dist['SELL']}/{dist['HOLD']}")
    print(f"  Win rate:       {metrics['win_rate_pct']:.1f}%")
    print(f"  Avg return:     {metrics['avg_return_per_trade_pct']:+.2f}%")
    print(f"  Portfolio:      $10,000 → ${metrics['portfolio_end_compound']:,.2f} "
          f"({metrics['total_return_compound_pct']:+.2f}%)")
    print(f"  Max drawdown:   {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio:   {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino ratio:  {metrics['sortino_ratio']:.3f}")
    print(f"  Buy&Hold:       {metrics['benchmark_buyhold_pct']:+.2f}%")
    print(f"  Alpha:          {metrics['alpha_vs_buyhold_pct']:+.2f}%")
    print(f"{'─'*60}")
    print(f"  → {output_file}")

    return payload


# ─── CLI ───────────────────────────────────────────────────────

_DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "JPM", "SPY", "QQQ"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-LLM baselines for ablation study")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), required=True)
    parser.add_argument("--tickers", nargs="+", default=_DEFAULT_TICKERS)
    parser.add_argument("--start", default="2025-01-06")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--hold-days", type=int, default=7)
    parser.add_argument("--freq-weeks", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/baseline_{args.strategy}.json"

    run_baseline(
        strategy_name=args.strategy,
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        hold_days=args.hold_days,
        freq_weeks=args.freq_weeks,
        output_file=args.output,
    )
