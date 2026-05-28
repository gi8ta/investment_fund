"""
yfinance fundamentals — point-in-time-safe fundamentals overview.

Unlike Alpha Vantage OVERVIEW (which always returns current values),
yfinance ticker.info returns current snapshot BUT we document this clearly
and complement it with the already-filtered balance sheet/CF/IS data.

This is the preferred fundamentals vendor for backtesting because the
financial statement endpoints (balance_sheet, cashflow, income_stmt) in
y_finance.py are filtered by curr_date.
"""

import yfinance as yf
from datetime import datetime


def get_fundamentals_overview(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve company fundamentals overview from yfinance.

    Note: ticker.info returns current-state data. For backtesting,
    the critical point-in-time data comes from the filtered financial
    statements (balance sheet, cash flow, income statement) which ARE
    filtered by curr_date in y_finance.py.

    Args:
        ticker    : Stock ticker symbol
        curr_date : Reference date (informational)

    Returns:
        Formatted string with key fundamentals.
    """
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
    except Exception as e:
        return f"Failed to retrieve fundamentals for {ticker}: {e}"

    if not info or info.get("regularMarketPrice") is None:
        return f"No fundamentals data found for {ticker}."

    fields = [
        ("Company", info.get("longName") or info.get("shortName", ticker)),
        ("Sector", info.get("sector")),
        ("Industry", info.get("industry")),
        ("Market Cap", _fmt_large_number(info.get("marketCap"))),
        ("Enterprise Value", _fmt_large_number(info.get("enterpriseValue"))),
        ("", None),  # separator
        ("P/E (Trailing)", _fmt_float(info.get("trailingPE"))),
        ("P/E (Forward)", _fmt_float(info.get("forwardPE"))),
        ("PEG Ratio", _fmt_float(info.get("pegRatio"))),
        ("P/S (Trailing)", _fmt_float(info.get("priceToSalesTrailing12Months"))),
        ("P/B", _fmt_float(info.get("priceToBook"))),
        ("EV/EBITDA", _fmt_float(info.get("enterpriseToEbitda"))),
        ("", None),  # separator
        ("EPS (Trailing)", _fmt_float(info.get("trailingEps"))),
        ("EPS (Forward)", _fmt_float(info.get("forwardEps"))),
        ("Revenue (TTM)", _fmt_large_number(info.get("totalRevenue"))),
        ("Revenue Growth", _fmt_pct(info.get("revenueGrowth"))),
        ("Earnings Growth", _fmt_pct(info.get("earningsGrowth"))),
        ("", None),  # separator
        ("Profit Margin", _fmt_pct(info.get("profitMargins"))),
        ("Gross Margin", _fmt_pct(info.get("grossMargins"))),
        ("Operating Margin", _fmt_pct(info.get("operatingMargins"))),
        ("ROE", _fmt_pct(info.get("returnOnEquity"))),
        ("ROA", _fmt_pct(info.get("returnOnAssets"))),
        ("", None),  # separator
        ("Debt/Equity", _fmt_float(info.get("debtToEquity"))),
        ("Current Ratio", _fmt_float(info.get("currentRatio"))),
        ("Quick Ratio", _fmt_float(info.get("quickRatio"))),
        ("Free Cash Flow", _fmt_large_number(info.get("freeCashflow"))),
        ("", None),  # separator
        ("Beta", _fmt_float(info.get("beta"))),
        ("52W High", _fmt_float(info.get("fiftyTwoWeekHigh"))),
        ("52W Low", _fmt_float(info.get("fiftyTwoWeekLow"))),
        ("50D MA", _fmt_float(info.get("fiftyDayAverage"))),
        ("200D MA", _fmt_float(info.get("twoHundredDayAverage"))),
        ("Dividend Yield", _fmt_pct(info.get("dividendYield"))),
    ]

    lines = [f"## {ticker.upper()} Fundamentals Overview\n"]
    for label, value in fields:
        if label == "" and value is None:
            continue  # skip separators with no preceding content
        if value is not None:
            lines.append(f"- **{label}:** {value}")

    if len(lines) <= 1:
        return f"No fundamentals data found for {ticker}."

    lines.append("")
    lines.append(
        f"_Note: Fundamentals are current snapshot values from Yahoo Finance. "
        f"Financial statements (BS/CF/IS) are separately filtered to {curr_date or 'latest'} "
        f"for point-in-time accuracy._"
    )

    return "\n".join(lines)


def _fmt_float(val):
    if val is None:
        return None
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_pct(val):
    if val is None:
        return None
    try:
        return f"{float(val)*100:.1f}%"
    except (ValueError, TypeError):
        return str(val)


def _fmt_large_number(val):
    if val is None:
        return None
    try:
        val = float(val)
        if abs(val) >= 1e12:
            return f"${val/1e12:.2f}T"
        elif abs(val) >= 1e9:
            return f"${val/1e9:.2f}B"
        elif abs(val) >= 1e6:
            return f"${val/1e6:.2f}M"
        else:
            return f"${val:,.0f}"
    except (ValueError, TypeError):
        return str(val)
