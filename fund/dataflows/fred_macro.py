"""
FRED (Federal Reserve Economic Data) — macroeconomic data vendor for Investment Fund.

Provides historical point-in-time macro indicators: CPI, unemployment, Fed Funds Rate,
Treasury yields, GDP, consumer sentiment, VIX.

Requires: FRED_API_KEY environment variable (free at https://fred.stlouisfed.org/docs/api/api_key.html).
Falls back to yfinance for VIX if FRED is unavailable.
"""

import os
from datetime import datetime, timedelta


def _get_fred():
    """Lazy-load fredapi to avoid import errors if not installed."""
    try:
        from fredapi import Fred
    except ImportError:
        raise RuntimeError(
            "fredapi is not installed. Run: pip install fredapi"
        )
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=key)


# Key macro series for trading decisions
MACRO_SERIES = {
    "CPIAUCSL":  ("CPI (All Urban Consumers)",       "monthly", "inflation"),
    "CPILFESL":  ("Core CPI (ex Food & Energy)",      "monthly", "inflation"),
    "FEDFUNDS":  ("Federal Funds Effective Rate",      "monthly", "rates"),
    "UNRATE":    ("Unemployment Rate",                 "monthly", "employment"),
    "PAYEMS":    ("Nonfarm Payrolls (thousands)",      "monthly", "employment"),
    "DGS10":     ("10-Year Treasury Yield",            "daily",   "rates"),
    "DGS2":      ("2-Year Treasury Yield",             "daily",   "rates"),
    "T10Y2Y":    ("10Y-2Y Spread (inversion signal)",  "daily",   "rates"),
    "UMCSENT":   ("U. Michigan Consumer Sentiment",    "monthly", "sentiment"),
    "VIXCLS":    ("VIX (CBOE Volatility Index)",       "daily",   "volatility"),
}


def get_macro_snapshot(curr_date: str, lookback_days: int = 90) -> str:
    """
    Retrieve a snapshot of key macroeconomic indicators as of curr_date.

    Returns formatted text with latest values for each macro series,
    suitable for injection into analyst prompts.
    """
    try:
        fred = _get_fred()
    except RuntimeError as e:
        return f"Macro data unavailable: {e}"

    dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    lines = [f"## Macroeconomic Snapshot (as of {curr_date})\n"]
    sections = {"inflation": [], "rates": [], "employment": [], "sentiment": [], "volatility": []}

    for series_id, (label, freq, category) in MACRO_SERIES.items():
        try:
            data = fred.get_series(series_id, observation_start=start, observation_end=curr_date)
            if data is not None and len(data) > 0:
                # Drop NaN values
                data = data.dropna()
                if len(data) == 0:
                    continue
                latest_val = data.iloc[-1]
                latest_date = data.index[-1].strftime("%Y-%m-%d")

                # Show trend if we have enough data
                trend_str = ""
                if len(data) >= 2:
                    prev_val = data.iloc[-2]
                    diff = latest_val - prev_val
                    if diff > 0:
                        trend_str = f" (↑ {diff:+.2f})"
                    elif diff < 0:
                        trend_str = f" (↓ {diff:+.2f})"
                    else:
                        trend_str = " (→ unchanged)"

                sections[category].append(
                    f"- **{label}:** {latest_val:.2f}{trend_str} ({latest_date})"
                )
        except Exception as e:
            # Skip individual series failures
            continue

    # Format by section
    section_labels = {
        "inflation": "### Inflation",
        "rates": "### Interest Rates & Yields",
        "employment": "### Employment",
        "sentiment": "### Consumer Sentiment",
        "volatility": "### Market Volatility",
    }

    for cat, label in section_labels.items():
        if sections[cat]:
            lines.append(label)
            lines.extend(sections[cat])
            lines.append("")

    if len(lines) <= 1:
        return f"No macro data available for the period ending {curr_date}."

    lines.append(
        "_Source: Federal Reserve Economic Data (FRED). "
        "Values are as-reported (point-in-time, no revisions applied)._"
    )
    return "\n".join(lines)


def get_vix_data(curr_date: str, lookback_days: int = 30) -> str:
    """
    Get VIX data. Tries FRED first, falls back to yfinance.
    """
    # Try FRED first
    try:
        fred = _get_fred()
        dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        data = fred.get_series("VIXCLS", observation_start=start, observation_end=curr_date)
        if data is not None and len(data.dropna()) > 0:
            data = data.dropna()
            latest = data.iloc[-1]
            latest_date = data.index[-1].strftime("%Y-%m-%d")

            # Interpret VIX level
            if latest > 30:
                regime = "EXTREME FEAR — elevated volatility, high risk"
            elif latest > 20:
                regime = "ELEVATED — above-average uncertainty"
            elif latest > 15:
                regime = "NORMAL — typical market conditions"
            else:
                regime = "COMPLACENCY — unusually low volatility, watch for surprises"

            lines = [
                f"## VIX (CBOE Volatility Index) as of {latest_date}",
                f"- **Current VIX:** {latest:.2f}",
                f"- **Regime:** {regime}",
                "",
            ]

            # Show recent trend
            if len(data) >= 5:
                last5 = data.tail(5)
                lines.append("**Recent VIX readings:**")
                for date, val in last5.items():
                    lines.append(f"  {date.strftime('%Y-%m-%d')}: {val:.2f}")

            return "\n".join(lines)
    except Exception:
        pass

    # Fallback to yfinance
    try:
        import yfinance as yf
        dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        vix = yf.download("^VIX", start=start, end=curr_date, progress=False)
        if vix is not None and not vix.empty:
            latest = float(vix["Close"].iloc[-1].item() if hasattr(vix["Close"].iloc[-1], 'item') else vix["Close"].iloc[-1])
            latest_date = vix.index[-1].strftime("%Y-%m-%d")

            if latest > 30:
                regime = "EXTREME FEAR"
            elif latest > 20:
                regime = "ELEVATED"
            elif latest > 15:
                regime = "NORMAL"
            else:
                regime = "COMPLACENCY"

            return (
                f"## VIX as of {latest_date}\n"
                f"- **Current VIX:** {latest:.2f}\n"
                f"- **Regime:** {regime}\n"
                f"_Source: Yahoo Finance_"
            )
    except Exception:
        pass

    return f"VIX data unavailable for {curr_date}."
