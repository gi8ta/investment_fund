#!/usr/bin/env python3
"""
Investment Fund Backtester

Runs propagate() across multiple tickers and weekly dates,
simulates a long/short portfolio, and outputs performance metrics.

Usage:
    python backtest.py
    python backtest.py --tickers NVDA AAPL MSFT --start 2026-02-02 --end 2026-02-28
    python backtest.py --max-workers 3   # run up to 3 tickers in parallel per date
    python backtest.py --help

What you see in the terminal:
    [  3/10]  NVDA  2026-02-02  (elapsed 04:21)
              → BUY  entry=136.40  exit=142.10  ret=+4.18%  ✓

All DEBUG/vendor noise goes to backtest_verbose.log (not the terminal).
backtest_results.json is updated incrementally after every signal.
"""

import os
import sys
import io
import json
import math
import logging
import time
import threading
import argparse
import contextlib
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from langchain_core.callbacks import BaseCallbackHandler

from fund.graph.trading_graph import TradingGraph
from fund.default_config import DEFAULT_CONFIG


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    ticker: str
    signal_date: str
    signal: str           # BUY / SELL / HOLD
    entry_price: float
    exit_date: str
    exit_price: float
    return_pct: float     # signed % return (accounts for direction)
    profitable: bool


# ─────────────────────────────────────────────────────────────
# Price data helpers (yfinance — no rate limit)
# ─────────────────────────────────────────────────────────────

def prefetch_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    buffer_days: int = 30,
    max_retries: int = 3,
) -> dict:
    end_ext = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    cache = {}
    for ticker in tickers:
        for attempt in range(1, max_retries + 1):
            try:
                df = yf.download(ticker, start=start_date, end=end_ext, progress=False, auto_adjust=True)
                if df.empty:
                    _log(f"  ⚠  {ticker}: no price data")
                    break
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                cache[ticker] = df[["Open", "Close"]].copy()
                _log(f"  ✓  {ticker}: {len(df)} trading days")
                break
            except Exception as exc:
                if attempt < max_retries:
                    _log(f"  ↺  {ticker}: download failed ({exc.__class__.__name__}), retrying in 5s ...")
                    time.sleep(5)
                else:
                    _log(f"  ✗  {ticker}: {exc}")
    return cache


def price_on_or_after(ticker: str, date: str, cache: dict, col: str = "Close") -> Optional[float]:
    if ticker not in cache:
        return None
    df = cache[ticker]
    available = df.index[df.index >= pd.Timestamp(date)]
    if available.empty:
        return None
    return float(df.loc[available[0], col])


def next_trading_day(date: str, n_calendar_days: int, ticker: str, cache: dict) -> str:
    target = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=n_calendar_days)
    if ticker not in cache:
        return target.strftime("%Y-%m-%d")
    df = cache[ticker]
    available = df.index[df.index >= pd.Timestamp(target)]
    if available.empty:
        return target.strftime("%Y-%m-%d")
    return str(available[0].date())


# ─────────────────────────────────────────────────────────────
# Date range generator
# ─────────────────────────────────────────────────────────────

def signal_dates(start: str, end: str, freq_weeks: int = 1) -> List[str]:
    cur  = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end,   "%Y-%m-%d")
    if freq_weeks <= 0:
        # Daily mode: every weekday (Mon-Fri)
        dates = []
        while cur <= stop:
            if cur.weekday() < 5:  # Mon=0 .. Fri=4
                dates.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
        return dates
    # Weekly mode: align to Monday, step by freq_weeks
    while cur.weekday() != 0:
        cur += timedelta(days=1)
    dates = []
    while cur <= stop:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(weeks=freq_weeks)
    return dates


# ─────────────────────────────────────────────────────────────
# Thread-local stdout/stderr router
#
# Replaces sys.stdout / sys.stderr with a proxy that routes each
# thread's writes to its own StringIO buffer (set via _suppress_to_file).
# Threads with no active buffer pass through to the real terminal.
# This makes _suppress_to_file() safe to call from multiple threads
# simultaneously without one thread stomping another's redirect.
# ─────────────────────────────────────────────────────────────

class _ThreadLocalStream:
    """Per-thread stream proxy. Writes go to the thread's buffer if set,
    otherwise to the original stream (terminal)."""

    def __init__(self, original):
        self._original = original
        self._local = threading.local()

    def write(self, s):
        buf = getattr(self._local, "buf", None)
        if buf is not None:
            buf.write(s)
        else:
            self._original.write(s)

    def flush(self):
        buf = getattr(self._local, "buf", None)
        if buf is None:
            self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._original, name)


# ─────────────────────────────────────────────────────────────
# LLM character-count callback
#
# Attached to every LLM at construction time (via config["callbacks"]).
# All prints go to the thread's suppressed buffer → log file only.
# ─────────────────────────────────────────────────────────────

class _CharCountCallback(BaseCallbackHandler):
    """Logs input/output character counts for every LLM call to the verbose log."""

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        total_chars = sum(
            len(str(m.content)) for batch in messages for m in batch
        )
        model_id = (serialized.get("kwargs") or {}).get("model_name", "?")
        print(
            f"[LLM→] {model_id}: {total_chars:,} chars "
            f"(~{total_chars // 4:,} tokens estimated)"
        )

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        total_chars = sum(
            len(gen.text)
            for gen_list in response.generations
            for gen in gen_list
            if hasattr(gen, "text")
        )
        print(f"[LLM←] {total_chars:,} chars response (~{total_chars // 4:,} tokens)")


# ─────────────────────────────────────────────────────────────
# Logging
#
# Two destinations:
#   - Terminal (INFO): clean progress lines, always via sys.__stdout__
#   - Log file (DEBUG): everything + vendor noise via _suppress_to_file
# ─────────────────────────────────────────────────────────────

_logger: logging.Logger = None
_log_fh  = None          # raw file handle
_tls_out: _ThreadLocalStream = None
_tls_err: _ThreadLocalStream = None
_log_file_lock = threading.Lock()


def _setup_logging(verbose_log: str):
    global _logger, _log_fh, _tls_out, _tls_err

    _log_fh = open(verbose_log, "w", encoding="utf-8", buffering=1)

    # Install thread-local stream wrappers so _suppress_to_file is thread-safe
    _tls_out = _ThreadLocalStream(sys.__stdout__)
    _tls_err = _ThreadLocalStream(sys.__stderr__)
    sys.stdout = _tls_out
    sys.stderr = _tls_err

    _logger = logging.getLogger("backtest")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()
    _logger.propagate = False

    # Console: INFO only, always writes to the REAL terminal (sys.__stdout__)
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(ch)

    # File: DEBUG with timestamps
    fh = logging.StreamHandler(_log_fh)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(fh)

    _logger.debug(f"=== Backtest verbose log — {datetime.now().isoformat()} ===")


def _teardown_logging():
    global _logger, _log_fh, _tls_out, _tls_err
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _tls_out = None
    _tls_err = None
    if _logger:
        for h in list(_logger.handlers):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
        _logger.handlers.clear()
        _logger = None
    if _log_fh:
        try:
            _log_fh.close()
        except Exception:
            pass
        _log_fh = None


def _log(msg: str):
    """INFO: visible on terminal and written to log file."""
    if _logger:
        _logger.info(msg)
    else:
        print(msg, flush=True)


def _dbg(msg: str):
    """DEBUG: log file only."""
    if _logger:
        _logger.debug(msg)


@contextlib.contextmanager
def _suppress_to_file():
    """Thread-safe: routes this thread's stdout/stderr to a per-thread buffer
    that is atomically written to the log file on exit."""
    if _log_fh is None or _tls_out is None:
        yield
        return
    buf = io.StringIO()
    _tls_out._local.buf = buf
    _tls_err._local.buf = buf
    try:
        yield
    finally:
        _tls_out._local.buf = None  # type: ignore[attr-defined]
        _tls_err._local.buf = None  # type: ignore[attr-defined]
        captured = buf.getvalue()
        if captured:
            with _log_file_lock:
                _log_fh.write(captured)
                _log_fh.flush()


# ─────────────────────────────────────────────────────────────
# Incremental results persistence
# ─────────────────────────────────────────────────────────────

def _save_partial_results(
    output_file: str,
    results: list,
    errors: list,
    config_snippet: dict,
    metrics: dict = None,
    monthly: dict = None,
    by_ticker: dict = None,
):
    """Write current results to JSON. Never raises."""
    try:
        payload = {
            "status": "complete" if metrics is not None else "in_progress",
            "as_of": datetime.now().isoformat(),
            "config": config_snippet,
            "trades": [asdict(r) for r in results],
            "errors": errors,
        }
        if metrics is not None:
            payload["metrics"] = metrics
        if monthly is not None:
            payload["monthly_breakdown"] = monthly
        if by_ticker is not None:
            payload["per_ticker_breakdown"] = by_ticker
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        _dbg(f"Warning: failed to save partial results: {exc}")


def _load_partial_results(output_file: str):
    """Resume support: load already-completed trades from a partial JSON.

    Returns (results, errors, done_keys) where done_keys is a set of
    (ticker, signal_date) pairs that have either been completed or
    permanently errored, so the caller can skip them.

    Silently returns ([], [], set()) if the file is missing, unreadable,
    or already marked complete.
    """
    if not os.path.exists(output_file):
        return [], [], set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") == "complete":
            return [], [], set()

        trades_raw = payload.get("trades") or []
        results = [TradeResult(**t) for t in trades_raw]

        errors = payload.get("errors") or []

        done = {(r.ticker, r.signal_date) for r in results}
        for e in errors:
            t = e.get("ticker"); d = e.get("date")
            if t and d:
                done.add((t, d))

        if results or errors:
            _log(f"\nResuming from {output_file}: "
                 f"{len(results)} prior trades, {len(errors)} prior errors. "
                 f"Skipping {len(done)} (ticker, date) pairs.")
        return results, errors, done
    except Exception as exc:
        _dbg(f"Warning: could not resume from {output_file}: {exc}")
        return [], [], set()


# ─────────────────────────────────────────────────────────────
# Parallel propagate worker
# ─────────────────────────────────────────────────────────────

_TRANSIENT_PATTERNS = (
    "429", "rate limit", "rate-limited", "too many requests",
    "temporarily", "timeout", "timed out", "connection reset",
    "connection error", "service unavailable",
    "502", "503", "504", "bad gateway", "internal server error",
)


def _propagate_worker(
    ta: TradingGraph,
    ticker: str,
    date: str,
    max_retries: int = 3,
    base_delay: int = 30,
):
    """Thread worker: calls ta.propagate() with retry and per-thread output capture.

    Returns (state, signal). Raises on unrecoverable failure.
    All stdout/stderr output is captured per-thread and written atomically to
    the log file — no interleaving even with multiple concurrent workers.
    """
    for attempt in range(1, max_retries + 1):
        try:
            with _suppress_to_file():
                state, signal = ta.propagate(ticker, date)
            return state, signal

        except Exception as exc:
            err_lower = str(exc).lower()
            is_transient = (
                any(p in err_lower for p in _TRANSIENT_PATTERNS)
                or isinstance(exc, (ConnectionError, TimeoutError, OSError))
            )
            if is_transient and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                _log(
                    f"         ↳ [{ticker}] transient error ({type(exc).__name__}) — "
                    f"retrying in {delay}s (attempt {attempt}/{max_retries}) ..."
                )
                _dbg(f"Retry reason: {exc}")
                time.sleep(delay)
            else:
                raise


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def calculate_metrics(
    results: List[TradeResult],
    tickers,
    cache,
    start,
    end,
    freq_weeks: int = 2,
) -> dict:
    active = [r for r in results if r.signal != "HOLD"]
    buy    = [r for r in results if r.signal == "BUY"]
    sell   = [r for r in results if r.signal == "SELL"]
    hold   = [r for r in results if r.signal == "HOLD"]

    n_tickers  = max(len(tickers), 1)
    trade_rets = [r.return_pct for r in active]
    win_rate   = (sum(1 for r in active if r.profitable) / len(active) * 100) if active else 0.0
    avg_ret    = sum(trade_rets) / len(trade_rets) if trade_rets else 0.0

    # ── Compound portfolio ────────────────────────────────────────────────────
    # Position = 1/n_tickers of CURRENT capital. Losses shrink future bets.
    capital     = 10_000.0
    equity_peak = capital
    max_dd_pct  = 0.0

    for r in sorted(results, key=lambda x: (x.signal_date, x.ticker)):
        if r.signal != "HOLD":
            position = capital / n_tickers
            capital += position * (r.return_pct / 100)
            capital  = max(capital, 0.0)
            if capital > equity_peak:
                equity_peak = capital
            dd = (equity_peak - capital) / equity_peak * 100
            max_dd_pct = max(max_dd_pct, dd)

    total_ret_compound = (capital - 10_000) / 10_000 * 100

    # ── Simple fixed-allocation reference ────────────────────────────────────
    simple_cap  = 10_000.0
    fixed_alloc = simple_cap / n_tickers
    for r in sorted(results, key=lambda x: x.signal_date):
        if r.signal != "HOLD":
            simple_cap += fixed_alloc * (r.return_pct / 100)
    total_ret_simple = (simple_cap - 10_000) / 10_000 * 100

    # ── Sharpe ratio (annualised, period-level) ───────────────────────────────
    # One portfolio return per signal date. Annualised with 52/freq_weeks.
    period_buckets: dict = defaultdict(list)
    for r in results:
        period_buckets[r.signal_date].append(
            r.return_pct if r.signal != "HOLD" else 0.0
        )
    period_rets = [
        sum(v + [0.0] * (n_tickers - len(v))) / n_tickers
        for v in [period_buckets[d] for d in sorted(period_buckets)]
    ]
    sharpe = 0.0
    if len(period_rets) >= 2:
        mean_p = sum(period_rets) / len(period_rets)
        std_p  = math.sqrt(
            sum((p - mean_p) ** 2 for p in period_rets) / (len(period_rets) - 1)
        )
        if std_p > 0:
            # Annualisation: daily → sqrt(252), weekly → sqrt(52/freq_weeks)
            ann_factor = math.sqrt(252) if freq_weeks <= 0 else math.sqrt(52.0 / freq_weeks)
            sharpe = round((mean_p / std_p) * ann_factor, 3)

    # ── Calmar ratio ──────────────────────────────────────────────────────────
    calmar = round(total_ret_compound / max_dd_pct, 3) if max_dd_pct > 0 else None

    # ── Buy-and-hold benchmark ────────────────────────────────────────────────
    bh_rets = []
    for t in tickers:
        ep = price_on_or_after(t, start, cache)
        xp = price_on_or_after(t, end,   cache)
        if ep and xp:
            bh_rets.append((xp - ep) / ep * 100)
    bh_ret = sum(bh_rets) / len(bh_rets) if bh_rets else 0.0

    return {
        "total_signals":               len(results),
        "active_signals":              len(active),
        "signal_distribution":         {"BUY": len(buy), "SELL": len(sell), "HOLD": len(hold)},
        "win_rate_pct":                round(win_rate, 2),
        "avg_return_per_trade_pct":    round(avg_ret, 4),
        "best_trade_pct":              round(max(trade_rets), 4) if trade_rets else 0,
        "worst_trade_pct":             round(min(trade_rets), 4) if trade_rets else 0,
        # Compound portfolio (primary)
        "portfolio_start":             10_000,
        "portfolio_end_compound":      round(capital, 2),
        "total_return_compound_pct":   round(total_ret_compound, 4),
        # Simple reference
        "portfolio_end_simple":        round(simple_cap, 2),
        "total_return_simple_pct":     round(total_ret_simple, 4),
        # Risk
        "max_drawdown_pct":            round(max_dd_pct, 2),
        "sharpe_ratio":                sharpe,
        "calmar_ratio":                calmar,
        # Benchmark
        "benchmark_buyhold_pct":       round(bh_ret, 4),
        "alpha_vs_buyhold_pct":        round(total_ret_compound - bh_ret, 4),
    }


def monthly_breakdown(results: List[TradeResult]) -> dict:
    """Per-calendar-month performance stats."""
    buckets: dict = defaultdict(list)
    for r in results:
        buckets[r.signal_date[:7]].append(r)   # "YYYY-MM"
    out = {}
    for month in sorted(buckets.keys()):
        rs     = buckets[month]
        active = [r for r in rs if r.signal != "HOLD"]
        wins   = sum(1 for r in active if r.profitable)
        rets   = [r.return_pct for r in active]
        out[month] = {
            "signals":       len(rs),
            "active":        len(active),
            "buy":           sum(1 for r in rs if r.signal == "BUY"),
            "sell":          sum(1 for r in rs if r.signal == "SELL"),
            "hold":          sum(1 for r in rs if r.signal == "HOLD"),
            "win_rate_pct":  round(wins / len(active) * 100, 1) if active else 0.0,
            "avg_return_pct": round(sum(rets) / len(rets), 3) if rets else 0.0,
            "total_return_pct": round(sum(rets), 3),
        }
    return out


def per_ticker_breakdown(results: List[TradeResult]) -> dict:
    """Per-ticker performance stats."""
    buckets: dict = defaultdict(list)
    for r in results:
        buckets[r.ticker].append(r)
    out = {}
    for ticker in sorted(buckets.keys()):
        rs     = buckets[ticker]
        active = [r for r in rs if r.signal != "HOLD"]
        wins   = sum(1 for r in active if r.profitable)
        rets   = [r.return_pct for r in active]
        out[ticker] = {
            "signals":        len(rs),
            "active":         len(active),
            "buy":            sum(1 for r in rs if r.signal == "BUY"),
            "sell":           sum(1 for r in rs if r.signal == "SELL"),
            "hold":           sum(1 for r in rs if r.signal == "HOLD"),
            "win_rate_pct":   round(wins / len(active) * 100, 1) if active else 0.0,
            "avg_return_pct": round(sum(rets) / len(rets), 3) if rets else 0.0,
            "best_pct":       round(max(rets), 3) if rets else 0.0,
            "worst_pct":      round(min(rets), 3) if rets else 0.0,
        }
    return out


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def print_summary(metrics: dict, config: dict, elapsed: float, errors: list,
                  monthly: dict = None, by_ticker: dict = None):
    sep = "=" * 68
    _log(f"\n{sep}")
    _log("  BACKTEST RESULTS SUMMARY")
    _log(sep)
    _log(f"  Model:             {config.get('deep_think_llm')}")
    _log(f"  Debate rounds:     {config.get('max_debate_rounds')}")
    _log(f"  Total runtime:     {_fmt_time(elapsed)}")
    _log(sep)
    _log(f"  Total signals:     {metrics['total_signals']}")
    _log(f"  Errors / skips:    {len(errors)}")
    _log(f"  Active (non-HOLD): {metrics['active_signals']}")
    dist = metrics['signal_distribution']
    _log(f"  BUY / SELL / HOLD: {dist['BUY']} / {dist['SELL']} / {dist['HOLD']}")
    _log(f"  Win rate:          {metrics['win_rate_pct']:.1f}%")
    _log(f"  Avg return/trade:  {metrics['avg_return_per_trade_pct']:+.2f}%")
    _log(f"  Best trade:        {metrics['best_trade_pct']:+.2f}%")
    _log(f"  Worst trade:       {metrics['worst_trade_pct']:+.2f}%")
    _log("-" * 68)
    _log("  PORTFOLIO  (compound: position = 1/N of current capital)")
    _log(f"  $10,000  →  ${metrics['portfolio_end_compound']:,.2f}  "
         f"({metrics['total_return_compound_pct']:+.2f}%)")
    _log(f"  simple fixed-alloc: ${metrics['portfolio_end_simple']:,.2f}  "
         f"({metrics['total_return_simple_pct']:+.2f}%)")
    _log("-" * 68)
    _log("  RISK METRICS")
    _log(f"  Max drawdown:      {metrics['max_drawdown_pct']:.2f}%")
    _log(f"  Sharpe ratio:      {metrics['sharpe_ratio']:.3f}  (annualised, period-level)")
    calmar = metrics.get("calmar_ratio")
    _log(f"  Calmar ratio:      {calmar:.3f}" if calmar is not None else
         "  Calmar ratio:      N/A  (no drawdown)")
    _log("-" * 68)
    _log("  BENCHMARK")
    _log(f"  Buy & Hold:        {metrics['benchmark_buyhold_pct']:+.2f}%")
    _log(f"  Alpha:             {metrics['alpha_vs_buyhold_pct']:+.2f}%")
    _log(sep)

    # ── Per-ticker breakdown ──────────────────────────────────────────────────
    if by_ticker:
        _log("\n  PER-TICKER BREAKDOWN")
        _log(f"  {'Ticker':<8} {'Signals':>7} {'Active':>6} {'B/S/H':>9} {'WinRate':>8} {'AvgRet':>8} {'Best':>8} {'Worst':>8}")
        _log("  " + "-" * 66)
        for t, s in sorted(by_ticker.items()):
            bsh = f"{s['buy']}/{s['sell']}/{s['hold']}"
            _log(f"  {t:<8} {s['signals']:>7} {s['active']:>6} {bsh:>9} "
                 f"{s['win_rate_pct']:>7.1f}% {s['avg_return_pct']:>+7.2f}% "
                 f"{s['best_pct']:>+7.2f}% {s['worst_pct']:>+7.2f}%")
        _log("")

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    if monthly:
        _log("  MONTHLY BREAKDOWN")
        _log(f"  {'Month':<9} {'Signals':>7} {'Active':>6} {'B/S/H':>9} {'WinRate':>8} {'AvgRet':>8} {'TotRet':>8}")
        _log("  " + "-" * 58)
        for m, s in sorted(monthly.items()):
            bsh = f"{s['buy']}/{s['sell']}/{s['hold']}"
            _log(f"  {m:<9} {s['signals']:>7} {s['active']:>6} {bsh:>9} "
                 f"{s['win_rate_pct']:>7.1f}% {s['avg_return_pct']:>+7.2f}% "
                 f"{s['total_return_pct']:>+7.2f}%")
        _log("")

    _log(sep)


# ─────────────────────────────────────────────────────────────
# Main backtest loop
# ─────────────────────────────────────────────────────────────

def run_backtest(
    tickers:       List[str],
    start_date:    str,
    end_date:      str,
    config:        dict,
    hold_days:     int   = 7,
    freq_weeks:    int   = 2,
    max_workers:   int   = 1,
    output_file:   str   = "backtest_results.json",
    verbose_log:   str   = "backtest_verbose.log",
    position_size: float = 1000.0,
    intraday:      bool  = False,
) -> dict:
    """Run the full backtest.

    max_workers > 1: all tickers for a given signal date run in parallel
    (each ticker gets its own TradingGraph instance with independent
    ChromaDB memory).  Dates are still processed sequentially so that memory
    from earlier dates is available for later ones.
    """
    _setup_logging(verbose_log)
    t0 = time.time()

    config_snippet = {
        "tickers":                 tickers,
        "start_date":              start_date,
        "end_date":                end_date,
        "hold_days":               hold_days,
        "freq_weeks":              freq_weeks,
        "max_workers":             max_workers,
        "llm_provider":            config.get("llm_provider"),
        "deep_think_llm":          config.get("deep_think_llm"),
        "quick_think_llm":         config.get("quick_think_llm"),
        "max_debate_rounds":       config.get("max_debate_rounds"),
        "max_risk_discuss_rounds": config.get("max_risk_discuss_rounds"),
    }

    try:
        _log(f"\nFetching price data ...")
        cache = prefetch_prices(tickers, start_date, end_date)

        dates = signal_dates(start_date, end_date, freq_weeks)
        total = len(tickers) * len(dates)

        freq_str = "daily (every trading day)" if freq_weeks <= 0 else f"every {freq_weeks} week(s)"
        hold_str = "intraday (Open→Close)" if intraday else f"{hold_days} calendar days"

        _log(f"\n{'─'*62}")
        _log(f"  Tickers:     {', '.join(tickers)}")
        _log(f"  Period:      {start_date}  →  {end_date}")
        _log(f"  Frequency:   {freq_str}  ({len(dates)} signal dates)")
        _log(f"  Hold:        {hold_str}")
        _log(f"  Workers:     {max_workers}  ({'parallel per date' if max_workers > 1 else 'sequential'})")
        _log(f"  Total runs:  {total} propagate() calls")
        _log(f"  Verbose log: {verbose_log}")
        _log(f"  Results:     {output_file}  (written after every signal)")
        _log(f"{'─'*62}")

        # One TradingGraph per ticker — independent ChromaDB memory
        _log(f"\nInitializing {len(tickers)} agent instance(s) ...")
        ta_instances: dict[str, TradingGraph] = {}
        for ticker in tickers:
            with _suppress_to_file():
                ta_instances[ticker] = TradingGraph(debug=False, config=config)
            _log(f"  ✓  {ticker}")

        # Resume support: pick up trades/errors already in the output file.
        results, errors, done_keys = _load_partial_results(output_file)
        completed = len(done_keys)

        for date in dates:
            _log(f"\n  ── {date} {'─' * 40}")

            # Pre-check entry prices; skip tickers with no data
            # Intraday: entry=Open, exit=Close same day
            # Normal:   entry=Close signal day, exit=Close after hold_days
            entry_col = "Open" if intraday else "Close"
            date_work = []
            for ticker in tickers:
                if (ticker, date) in done_keys:
                    continue  # already done in a previous run
                entry_price = price_on_or_after(ticker, date, cache, col=entry_col)
                if entry_price is None:
                    completed += 1
                    _log(f"[{completed:>3}/{total}]  {ticker}  {date}  → SKIP: no entry price")
                    errors.append({"ticker": ticker, "date": date, "reason": "no_entry_price"})
                    done_keys.add((ticker, date))
                    _save_partial_results(output_file, results, errors, config_snippet)
                else:
                    date_work.append((ticker, entry_price))

            if not date_work:
                continue

            # ── Parallel execution for this date ──────────────────────────
            workers = min(max_workers, len(date_work))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_propagate_worker, ta_instances[t], t, date): (t, ep)
                    for t, ep in date_work
                }

                for future in as_completed(futures):
                    ticker, entry_price = futures[future]
                    elapsed = time.time() - t0
                    completed += 1

                    _log(f"[{completed:>3}/{total}]  {ticker}  {date}  "
                         f"(elapsed {_fmt_time(elapsed)})")

                    try:
                        # Hard ceiling per-ticker: even with LLM timeouts in
                        # place, this is a backstop against any single trade
                        # blocking the whole backtest forever.
                        _state, raw_signal = future.result(timeout=600)
                    except FuturesTimeoutError:
                        _log(f"         → TIMEOUT after 600s — skipping {ticker} {date}")
                        future.cancel()
                        errors.append({"ticker": ticker, "date": date, "reason": "propagate_timeout_600s"})
                        _save_partial_results(output_file, results, errors, config_snippet)
                        continue
                    except Exception as exc:
                        _log(f"         → ERROR: {exc}")
                        _dbg(f"TRACEBACK:\n{traceback.format_exc()}")
                        errors.append({"ticker": ticker, "date": date, "reason": str(exc)})
                        _save_partial_results(output_file, results, errors, config_snippet)
                        continue

                    s = raw_signal.strip().upper()
                    # CRO override takes precedence — if CRO rejected, it's HOLD
                    if "CRO OVERRIDE TO HOLD" in s:
                        signal = "HOLD"
                    else:
                        signal = "BUY" if "BUY" in s else "SELL" if "SELL" in s else "HOLD"

                    if intraday:
                        exit_date  = date  # same day
                        exit_price = price_on_or_after(ticker, date, cache, col="Close")
                    else:
                        exit_date  = next_trading_day(date, hold_days, ticker, cache)
                        exit_price = price_on_or_after(ticker, exit_date, cache)

                    if exit_price is None:
                        _log(f"         → SKIP: no exit price @ {exit_date}")
                        errors.append({"ticker": ticker, "date": date,
                                       "reason": f"no_exit_price_{exit_date}"})
                        _save_partial_results(output_file, results, errors, config_snippet)
                        continue

                    raw_ret = (exit_price - entry_price) / entry_price
                    return_pct = (raw_ret * 100 if signal == "BUY"
                                  else -raw_ret * 100 if signal == "SELL"
                                  else 0.0)

                    profitable = return_pct > 0
                    mark = "✓" if profitable else "✗"

                    results.append(TradeResult(
                        ticker=ticker,
                        signal_date=date,
                        signal=signal,
                        entry_price=round(entry_price, 4),
                        exit_date=exit_date,
                        exit_price=round(exit_price, 4),
                        return_pct=round(return_pct, 4),
                        profitable=profitable,
                    ))

                    _log(f"         → {signal:4s}  "
                         f"entry={entry_price:.2f}  exit={exit_price:.2f}  "
                         f"ret={return_pct:+.2f}%  {mark}")

                    _save_partial_results(output_file, results, errors, config_snippet)

                    if signal != "HOLD":
                        try:
                            dollar_pnl = position_size * (return_pct / 100)
                            with _suppress_to_file():
                                ta_instances[ticker].reflect_and_remember(dollar_pnl)
                        except Exception as exc:
                            _log(f"         ↳ reflect failed: {exc}")

                    # Update Darwinian weights (if enabled)
                    try:
                        with _suppress_to_file():
                            ta_instances[ticker].update_darwinian_weights(
                                ticker, date, signal, return_pct
                            )
                    except Exception as exc:
                        _dbg(f"         ↳ darwinian update failed: {exc}")

        elapsed_total = time.time() - t0
        metrics   = calculate_metrics(results, tickers, cache, start_date, end_date, freq_weeks)
        monthly   = monthly_breakdown(results)
        by_ticker = per_ticker_breakdown(results)

        _save_partial_results(output_file, results, errors, config_snippet,
                              metrics, monthly, by_ticker)

        _log(f"\n  Results → {output_file}")
        _log(f"  Verbose → {verbose_log}")
        print_summary(metrics, config, elapsed_total, errors, monthly, by_ticker)

        return {
            "status": "complete",
            "config": config_snippet,
            "trades": [asdict(r) for r in results],
            "errors": errors,
            "metrics": metrics,
            "monthly_breakdown": monthly,
            "per_ticker_breakdown": by_ticker,
        }

    finally:
        _teardown_logging()


# ─────────────────────────────────────────────────────────────
# JANUS multi-cohort backtest
# ─────────────────────────────────────────────────────────────

def run_janus_backtest(
    tickers:       List[str],
    start_date:    str,
    end_date:      str,
    config:        dict,
    hold_days:     int   = 7,
    freq_weeks:    int   = 2,
    output_file:   str   = "backtest_results.json",
    verbose_log:   str   = "backtest_verbose.log",
    position_size: float = 1000.0,
) -> dict:
    """Run JANUS multi-cohort backtest.

    For each signal date/ticker, runs propagate() once per cohort, then
    blends their signals via JanusMetaWeighter to produce a single trade.
    """
    from fund.meta.janus import JanusMetaWeighter

    cohort_configs = config.get("janus_cohorts", [])
    if not cohort_configs:
        raise ValueError("JANUS enabled but no cohorts configured")

    _setup_logging(verbose_log)
    t0 = time.time()

    janus = JanusMetaWeighter(
        cohort_configs=cohort_configs,
        state_file=config.get("janus_state_file", "janus_state.json"),
    )

    config_snippet = {
        "mode":                    "janus",
        "tickers":                 tickers,
        "start_date":              start_date,
        "end_date":                end_date,
        "hold_days":               hold_days,
        "freq_weeks":              freq_weeks,
        "cohorts":                 [c.name for c in cohort_configs],
        "llm_provider":            config.get("llm_provider"),
        "deep_think_llm":         config.get("deep_think_llm"),
        "quick_think_llm":        config.get("quick_think_llm"),
    }

    try:
        _log(f"\nFetching price data ...")
        cache = prefetch_prices(tickers, start_date, end_date)

        dates = signal_dates(start_date, end_date, freq_weeks)
        total = len(tickers) * len(dates)

        _log(f"\n{'─'*62}")
        _log(f"  Mode:        JANUS ({len(cohort_configs)} cohorts: {[c.name for c in cohort_configs]})")
        _log(f"  Tickers:     {', '.join(tickers)}")
        _log(f"  Period:      {start_date}  →  {end_date}")
        _log(f"  Frequency:   every {freq_weeks} week(s)  ({len(dates)} signal dates)")
        _log(f"  Hold:        {hold_days} calendar days")
        _log(f"  LLM calls:   ~{total * len(cohort_configs)} propagate() calls")
        _log(f"{'─'*62}")

        # Create one TradingGraph per (ticker, cohort)
        _log(f"\nInitializing {len(tickers)} x {len(cohort_configs)} agent instances ...")
        ta_cohorts: dict = {}  # (ticker, cohort_name) -> TradingGraph
        for ticker in tickers:
            for cc in cohort_configs:
                cohort_config = janus.get_cohort_config(config, cc.name)
                with _suppress_to_file():
                    ta_cohorts[(ticker, cc.name)] = TradingGraph(
                        debug=False, config=cohort_config
                    )
            _log(f"  ✓  {ticker} ({len(cohort_configs)} cohorts)")

        # Resume support: pick up trades/errors already in the output file.
        results, errors, done_keys = _load_partial_results(output_file)
        completed = len(done_keys)

        for date in dates:
            _log(f"\n  ── {date} {'─' * 40}")

            for ticker in tickers:
                if (ticker, date) in done_keys:
                    continue
                entry_price = price_on_or_after(ticker, date, cache)
                if entry_price is None:
                    completed += 1
                    _log(f"[{completed:>3}/{total}]  {ticker}  {date}  → SKIP: no entry price")
                    errors.append({"ticker": ticker, "date": date, "reason": "no_entry_price"})
                    done_keys.add((ticker, date))
                    continue

                # Run all cohorts for this ticker/date
                cohort_signals = {}
                for cc in cohort_configs:
                    ta = ta_cohorts[(ticker, cc.name)]
                    try:
                        with _suppress_to_file():
                            _state, raw_signal = ta.propagate(ticker, date)
                        s = raw_signal.strip().upper()
                        sig = "BUY" if "BUY" in s else "SELL" if "SELL" in s else "HOLD"
                        cohort_signals[cc.name] = sig
                        janus.record_signal(cc.name, ticker, date, sig)
                        _log(f"         cohort={cc.name} → {sig}")
                    except Exception as exc:
                        _log(f"         cohort={cc.name} → ERROR: {exc}")
                        cohort_signals[cc.name] = "HOLD"

                # Blend signals
                signal, votes = janus.blend_signals(cohort_signals)
                completed += 1

                elapsed = time.time() - t0
                _log(f"[{completed:>3}/{total}]  {ticker}  {date}  "
                     f"(elapsed {_fmt_time(elapsed)}) → JANUS blend: {signal}  votes={votes}")

                exit_date  = next_trading_day(date, hold_days, ticker, cache)
                exit_price = price_on_or_after(ticker, exit_date, cache)

                if exit_price is None:
                    _log(f"         → SKIP: no exit price @ {exit_date}")
                    errors.append({"ticker": ticker, "date": date,
                                   "reason": f"no_exit_price_{exit_date}"})
                    continue

                raw_ret = (exit_price - entry_price) / entry_price
                return_pct = (raw_ret * 100 if signal == "BUY"
                              else -raw_ret * 100 if signal == "SELL"
                              else 0.0)

                profitable = return_pct > 0
                mark = "✓" if profitable else "✗"

                results.append(TradeResult(
                    ticker=ticker,
                    signal_date=date,
                    signal=signal,
                    entry_price=round(entry_price, 4),
                    exit_date=exit_date,
                    exit_price=round(exit_price, 4),
                    return_pct=round(return_pct, 4),
                    profitable=profitable,
                ))

                _log(f"         → {signal:4s}  "
                     f"entry={entry_price:.2f}  exit={exit_price:.2f}  "
                     f"ret={return_pct:+.2f}%  {mark}")

                _save_partial_results(output_file, results, errors, config_snippet)

                # Update JANUS outcomes and weights
                for cc in cohort_configs:
                    janus.record_outcome(cc.name, ticker, date, return_pct)
                janus.update_weights()

        elapsed_total = time.time() - t0
        metrics   = calculate_metrics(results, tickers, cache, start_date, end_date, freq_weeks)
        monthly   = monthly_breakdown(results)
        by_ticker = per_ticker_breakdown(results)

        _save_partial_results(output_file, results, errors, config_snippet,
                              metrics, monthly, by_ticker)

        _log(f"\n  Results → {output_file}")
        _log(f"  Verbose → {verbose_log}")
        print_summary(metrics, config, elapsed_total, errors, monthly, by_ticker)

        return {
            "status": "complete",
            "mode": "janus",
            "config": config_snippet,
            "trades": [asdict(r) for r in results],
            "errors": errors,
            "metrics": metrics,
            "janus_weights": janus.weights,
        }

    finally:
        _teardown_logging()


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

_DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "JPM", "SPY", "QQQ"]

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Investment Fund Backtester")
    parser.add_argument("--tickers",       nargs="+", default=_DEFAULT_TICKERS)
    parser.add_argument("--start",         default="2025-01-06",
                        help="Start date (default: first Monday of 2025)")
    parser.add_argument("--end",           default="2025-12-29",
                        help="End date (default: last Monday of 2025)")
    parser.add_argument("--hold-days",     type=int,   default=7,
                        help="Calendar days to hold position (default 7 = 1 week)")
    parser.add_argument("--intraday",      action="store_true",
                        help="Intraday mode: enter at Open, exit at Close same day")
    parser.add_argument("--freq-weeks",    type=int,   default=1,
                        help="Signal frequency in weeks (default 1 = weekly)")
    parser.add_argument("--max-workers",   type=int,   default=3,
                        help="Tickers to run in parallel per signal date (default 3)")
    parser.add_argument("--debate-rounds", type=int,   default=1)
    parser.add_argument("--risk-rounds",   type=int,   default=1,
                        help="Risk debate rounds (default 1)")
    parser.add_argument("--provider",      default="openrouter")
    parser.add_argument("--deep-model",    default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--quick-model",   default="google/gemini-3.1-flash-lite-preview")
    parser.add_argument("--backend-url",   default="https://openrouter.ai/api/v1")
    parser.add_argument("--output",        default="backtest_results.json")
    parser.add_argument("--verbose-log",   default="backtest_verbose.log")
    parser.add_argument("--position-size", type=float, default=1000.0)
    parser.add_argument("--no-char-count", action="store_true",
                        help="Disable LLM character-count logging (slightly faster)")
    # ATLAS-GIC integration flags
    parser.add_argument("--enable-darwinian", action="store_true",
                        help="Enable Darwinian weight tracking for agents")
    parser.add_argument("--enable-cro", action="store_true",
                        help="Enable CRO adversarial review node")
    parser.add_argument("--cro-threshold", type=int, default=24,
                        help="CRO rejection threshold (6-30). Lower = more permissive. Default: 24")
    parser.add_argument("--enable-forward-context", action="store_true",
                        help="Enable forward context injection (catalysts/events)")
    parser.add_argument("--enable-autoresearch", action="store_true",
                        help="Enable autoresearch prompt optimization loop")
    parser.add_argument("--prompts-dir", default=None,
                        help="Directory for external prompt templates (default: built-in)")
    parser.add_argument("--enable-janus", action="store_true",
                        help="Enable JANUS meta-weighting (runs multiple config cohorts)")
    parser.add_argument("--janus-cohorts", default="aggressive,balanced",
                        help="Comma-separated cohort names (default: aggressive,balanced)")
    # Ablation study flags
    parser.add_argument("--skip-invest-debate", action="store_true",
                        help="Ablation: skip Bull/Bear debate (analysts → Research Manager directly)")
    parser.add_argument("--skip-risk-debate", action="store_true",
                        help="Ablation: skip Risky/Safe/Neutral debate (Trader → Risk Judge directly)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"]            = args.provider
    config["deep_think_llm"]          = args.deep_model
    config["quick_think_llm"]         = args.quick_model
    config["backend_url"]             = args.backend_url
    config["max_debate_rounds"]       = args.debate_rounds
    config["max_risk_discuss_rounds"] = args.risk_rounds
    # ATLAS-GIC integrations
    config["enable_darwinian_weights"] = args.enable_darwinian
    config["enable_cro"]               = args.enable_cro
    config["cro_rejection_threshold"]  = args.cro_threshold
    config["enable_forward_context"]   = args.enable_forward_context
    config["enable_autoresearch"]      = args.enable_autoresearch
    config["enable_janus"]             = args.enable_janus
    config["skip_invest_debate"]       = args.skip_invest_debate
    config["skip_risk_debate"]         = args.skip_risk_debate
    if args.prompts_dir:
        config["prompts_dir"]          = args.prompts_dir
    if args.enable_janus:
        from fund.meta.janus import CohortConfig
        cohort_names = [c.strip() for c in args.janus_cohorts.split(",")]
        _JANUS_DEFAULTS = {
            "aggressive": {"max_debate_rounds": 3, "max_risk_discuss_rounds": 2},
            "balanced":   {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1},
            "cautious":   {"max_debate_rounds": 2, "max_risk_discuss_rounds": 2, "enable_cro": True},
        }
        config["janus_cohorts"] = [
            CohortConfig(name=n, overrides=_JANUS_DEFAULTS.get(n, {}))
            for n in cohort_names
        ]
    # data_vendors: use defaults from DEFAULT_CONFIG (yfinance fundamentals, finnhub news)
    # No override needed — DEFAULT_CONFIG already has the correct point-in-time safe vendors
    # LLM character-count callback (writes to verbose log only)
    if not args.no_char_count:
        config["callbacks"] = [_CharCountCallback()]

    # Intraday mode: force daily frequency
    _freq = 0 if args.intraday else args.freq_weeks

    if config.get("enable_janus", False):
        run_janus_backtest(
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            config=config,
            hold_days=args.hold_days,
            freq_weeks=_freq,
            output_file=args.output,
            verbose_log=args.verbose_log,
            position_size=args.position_size,
        )
    else:
        run_backtest(
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            config=config,
            hold_days=args.hold_days,
            freq_weeks=_freq,
            max_workers=args.max_workers,
            output_file=args.output,
            verbose_log=args.verbose_log,
            position_size=args.position_size,
            intraday=args.intraday,
        )
