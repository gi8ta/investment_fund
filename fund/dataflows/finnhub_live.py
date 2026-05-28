"""
Finnhub Live API — real-time data vendor for Investment Fund.

Provides:
  - get_company_news()       : recent news for a specific ticker
  - get_market_news()        : general macro/market news
  - get_insider_sentiment()  : monthly insider buying/selling (MSPR)
  - get_basic_financials()   : key financial ratios (P/E, EPS, beta, etc.)
  - get_earnings_surprise()  : quarterly EPS actual vs estimate
  - get_analyst_ratings()    : analyst recommendation trends
  - get_social_sentiment()   : social media sentiment (Reddit, Twitter)

Requires: FINNHUB_API_KEY environment variable.
Free tier: 60 requests/minute — sufficient for all Investment Fund use cases.
"""

import os
from datetime import datetime, timedelta

import requests

_BASE_URL = "https://finnhub.io/api/v1"
_TIMEOUT = 15  # seconds


def _api_key() -> str:
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FINNHUB_API_KEY is not set. "
            "Get a free key at https://finnhub.io and add it to your .env file."
        )
    return key


def _get(endpoint: str, params: dict) -> dict:
    """Make a GET request to Finnhub API, raise on HTTP errors."""
    params["token"] = _api_key()
    url = f"{_BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────

def get_company_news(ticker: str, start_date: str, end_date: str) -> str:
    """
    Retrieve recent news articles for a specific ticker from Finnhub.

    Args:
        ticker     : Stock ticker symbol (e.g. "AAPL")
        start_date : Start date in yyyy-mm-dd format
        end_date   : End date in yyyy-mm-dd format

    Returns:
        Formatted string with headlines and summaries.
    """
    data = _get("company-news", {
        "symbol": ticker,
        "from": start_date,
        "to": end_date,
    })

    if not data:
        return f"No news found for {ticker} from {start_date} to {end_date}."

    lines = [f"## {ticker} News ({start_date} → {end_date})\n"]
    for article in data[:30]:  # cap at 30 articles
        dt = datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d")
        headline = article.get("headline", "").strip()
        summary  = article.get("summary",  "").strip()
        source   = article.get("source",   "")
        url      = article.get("url",      "")
        lines.append(f"### [{headline}]({url}) — {source} ({dt})")
        if summary:
            lines.append(summary)
        lines.append("")

    return "\n".join(lines)


def get_market_news(curr_date: str, look_back_days: int = 7, limit: int = 10) -> str:
    """
    Retrieve general market / macro news from Finnhub.

    Args:
        curr_date      : Reference date in yyyy-mm-dd format
        look_back_days : How many days back to look (default 7)
        limit          : Maximum number of articles to return (default 10)

    Returns:
        Formatted string with macro news headlines and summaries.
    """
    # Finnhub general news endpoint returns the latest N articles by category.
    # We filter by date client-side.
    cutoff_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    cutoff_ts = cutoff_dt.timestamp()

    data = _get("news", {"category": "general"})

    if not data:
        return f"No global market news available for the period ending {curr_date}."

    # Filter to the requested window
    filtered = [
        a for a in data
        if a.get("datetime", 0) >= cutoff_ts
    ][:limit]

    if not filtered:
        return f"No global market news found in the last {look_back_days} days before {curr_date}."

    lines = [f"## Global Market News (last {look_back_days} days before {curr_date})\n"]
    for article in filtered:
        dt = datetime.fromtimestamp(article.get("datetime", 0)).strftime("%Y-%m-%d")
        headline = article.get("headline", "").strip()
        summary  = article.get("summary",  "").strip()
        source   = article.get("source",   "")
        lines.append(f"### {headline} — {source} ({dt})")
        if summary:
            lines.append(summary)
        lines.append("")

    return "\n".join(lines)


def get_insider_sentiment(ticker: str, curr_date: str) -> str:
    """
    Retrieve monthly insider sentiment (MSPR) for a ticker from Finnhub.

    MSPR (Monthly Share Purchase Ratio) = purchases / (purchases + sales).
    Values near 1.0 = strong insider buying; near 0.0 = heavy insider selling.

    Args:
        ticker    : Stock ticker symbol
        curr_date : Reference date in yyyy-mm-dd format (looks back 3 months)

    Returns:
        Formatted string with monthly insider sentiment data.
    """
    dt      = datetime.strptime(curr_date, "%Y-%m-%d")
    from_dt = (dt - timedelta(days=90)).strftime("%Y-%m-%d")

    data = _get("stock/insider-sentiment", {
        "symbol": ticker,
        "from": from_dt,
        "to": curr_date,
    })

    entries = data.get("data", [])
    if not entries:
        return f"No insider sentiment data available for {ticker} (last 90 days before {curr_date})."

    lines = [
        f"## {ticker} Insider Sentiment ({from_dt} → {curr_date})\n",
        "| Year | Month | Change (shares) | MSPR |\n|------|-------|-----------------|------|",
    ]
    for e in entries:
        mspr   = e.get("mspr", 0)
        change = e.get("change", 0)
        lines.append(f"| {e.get('year')} | {e.get('month')} | {change:+,} | {mspr:.4f} |")

    lines.append("")
    lines.append(
        "**MSPR interpretation:** >0.6 = net insider buying (bullish signal), "
        "<0.4 = net insider selling (bearish signal). "
        "Change = net share count delta across all insider transactions."
    )
    return "\n".join(lines)


def get_basic_financials(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve key financial metrics for a ticker from Finnhub.

    Covers: P/E, P/B, EPS (TTM), ROE, ROA, debt/equity, current ratio,
    52-week high/low, beta, revenue per share, and more.

    Args:
        ticker    : Stock ticker symbol
        curr_date : Unused (included for vendor interface compatibility)

    Returns:
        Formatted string with key financial ratios.
    """
    data = _get("stock/metric", {"symbol": ticker, "metric": "all"})

    metric   = data.get("metric", {})
    series   = data.get("series", {})  # quarterly/annual time series (not used here)

    if not metric:
        return f"No fundamental financial data available for {ticker} via Finnhub."

    # Select the most useful fields for trading decisions
    fields = [
        ("P/E Ratio (TTM)",              metric.get("peTTM")),
        ("P/E Ratio (Annual)",           metric.get("peAnnual")),
        ("P/B Ratio",                    metric.get("pb")),
        ("P/S Ratio (TTM)",              metric.get("psTTM")),
        ("EPS (TTM)",                    metric.get("epsTTM")),
        ("EPS Growth (TTM vs prev TTM)", metric.get("epsGrowthTTMYoy")),
        ("Revenue Per Share (TTM)",      metric.get("revenuePerShareTTM")),
        ("Revenue Growth (YoY)",         metric.get("revenueGrowthTTMYoy")),
        ("ROE (TTM)",                    metric.get("roeTTM")),
        ("ROA (TTM)",                    metric.get("roaTTM")),
        ("Net Margin (TTM)",             metric.get("netMarginTTM")),
        ("Gross Margin (TTM)",           metric.get("grossMarginTTM")),
        ("Debt / Equity",                metric.get("totalDebt/totalEquityAnnual")),
        ("Current Ratio (Annual)",       metric.get("currentRatioAnnual")),
        ("Beta",                         metric.get("beta")),
        ("52W High",                     metric.get("52WeekHigh")),
        ("52W Low",                      metric.get("52WeekLow")),
        ("52W Return (%)",               metric.get("52WeekPriceReturnDaily")),
        ("Dividend Yield (%)",           metric.get("dividendYieldIndicatedAnnual")),
        ("Market Cap",                   metric.get("marketCapitalization")),
    ]

    lines = [f"## {ticker} Key Financial Metrics (Finnhub)\n"]
    for label, value in fields:
        if value is not None:
            lines.append(f"- **{label}:** {value}")

    if len(lines) == 1:
        return f"No key metrics returned for {ticker}."

    return "\n".join(lines)


def get_earnings_surprise(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve quarterly earnings surprise data (actual vs estimate EPS).

    Args:
        ticker    : Stock ticker symbol
        curr_date : Reference date (used to filter out future earnings)

    Returns:
        Formatted string with earnings surprise history.
    """
    data = _get("stock/earnings", {"symbol": ticker, "limit": 8})

    if not data:
        return f"No earnings data available for {ticker}."

    # Filter to only include earnings on or before curr_date
    if curr_date:
        data = [e for e in data if e.get("period", "9999-99-99") <= curr_date]

    if not data:
        return f"No earnings data available for {ticker} on or before {curr_date}."

    lines = [
        f"## {ticker} Earnings History (last {len(data)} quarters)\n",
        "| Period | Actual EPS | Estimate EPS | Surprise | Surprise % |",
        "|--------|-----------|-------------|----------|------------|",
    ]

    beat_count = 0
    for e in data:
        actual   = e.get("actual")
        estimate = e.get("estimate")
        period   = e.get("period", "N/A")
        surprise = e.get("surprise", 0)
        surprise_pct = e.get("surprisePercent", 0)

        actual_str   = f"{actual:.2f}" if actual is not None else "N/A"
        estimate_str = f"{estimate:.2f}" if estimate is not None else "N/A"
        surprise_str = f"{surprise:+.2f}" if surprise is not None else "N/A"
        pct_str      = f"{surprise_pct:+.1f}%" if surprise_pct is not None else "N/A"

        if surprise is not None and surprise > 0:
            beat_count += 1

        lines.append(f"| {period} | {actual_str} | {estimate_str} | {surprise_str} | {pct_str} |")

    lines.append("")
    total = len(data)
    if total > 0:
        lines.append(
            f"**Beat rate:** {beat_count}/{total} quarters ({beat_count/total*100:.0f}%). "
            f"Consistent beats suggest strong execution; misses may signal deterioration."
        )

    return "\n".join(lines)


def get_analyst_ratings(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve analyst recommendation trends from Finnhub.

    Returns monthly snapshots of buy/hold/sell consensus.

    Args:
        ticker    : Stock ticker symbol
        curr_date : Reference date (filters out future recommendations)

    Returns:
        Formatted string with analyst consensus trend.
    """
    data = _get("stock/recommendation", {"symbol": ticker})

    if not data:
        return f"No analyst recommendation data available for {ticker}."

    # Filter to only periods on or before curr_date
    if curr_date:
        data = [r for r in data if r.get("period", "9999-99-99") <= curr_date]

    if not data:
        return f"No analyst recommendations for {ticker} on or before {curr_date}."

    # Sort by period descending, take last 6 months
    data = sorted(data, key=lambda r: r.get("period", ""), reverse=True)[:6]

    lines = [
        f"## {ticker} Analyst Recommendations (last {len(data)} months)\n",
        "| Period | Strong Buy | Buy | Hold | Sell | Strong Sell | Total |",
        "|--------|-----------|-----|------|------|-------------|-------|",
    ]

    for r in data:
        sb = r.get("strongBuy", 0)
        b  = r.get("buy", 0)
        h  = r.get("hold", 0)
        s  = r.get("sell", 0)
        ss = r.get("strongSell", 0)
        total = sb + b + h + s + ss
        lines.append(f"| {r.get('period', 'N/A')} | {sb} | {b} | {h} | {s} | {ss} | {total} |")

    # Summary of latest period
    if data:
        latest = data[0]
        sb = latest.get("strongBuy", 0) + latest.get("buy", 0)
        bearish = latest.get("sell", 0) + latest.get("strongSell", 0)
        total = sb + bearish + latest.get("hold", 0)
        if total > 0:
            bull_pct = sb / total * 100
            lines.append("")
            if bull_pct >= 70:
                lines.append(f"**Consensus: BULLISH** ({bull_pct:.0f}% buy/strong-buy)")
            elif bull_pct >= 50:
                lines.append(f"**Consensus: MODERATELY BULLISH** ({bull_pct:.0f}% buy/strong-buy)")
            elif bull_pct >= 30:
                lines.append(f"**Consensus: MIXED** ({bull_pct:.0f}% buy/strong-buy)")
            else:
                lines.append(f"**Consensus: BEARISH** ({bull_pct:.0f}% buy/strong-buy)")

    return "\n".join(lines)


def get_social_sentiment(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve social media sentiment from Finnhub (Reddit + Twitter).

    Replaces StockTwits as the social sentiment source — Finnhub provides
    aggregated mention counts and sentiment scores from multiple platforms.

    Args:
        ticker    : Stock ticker symbol
        curr_date : Reference date (informational)

    Returns:
        Formatted string with social sentiment data.
    """
    try:
        data = _get("stock/social-sentiment", {"symbol": ticker})
    except Exception:
        # social-sentiment may not be available on free tier for all tickers
        data = {}

    reddit_data = data.get("reddit", [])
    twitter_data = data.get("twitter", [])

    if not reddit_data and not twitter_data:
        # Fallback: try buzz endpoint for social volume
        try:
            buzz = _get("stock/buzz", {"symbol": ticker})
            if buzz:
                lines = [
                    f"## {ticker} Social Buzz (as of {curr_date})",
                    f"- **Articles in last week:** {buzz.get('articlesInLastWeek', 'N/A')}",
                    f"- **Buzz:** {buzz.get('buzz', 'N/A')}",
                    f"- **Weekly Average:** {buzz.get('weeklyAverage', 'N/A')}",
                ]
                return "\n".join(lines)
        except Exception:
            pass
        return f"No social sentiment data available for {ticker}."

    lines = [f"## {ticker} Social Media Sentiment (as of {curr_date})\n"]

    # Process Reddit data
    if reddit_data:
        total_mentions = sum(d.get("mention", 0) for d in reddit_data[-24:])
        pos = sum(d.get("positiveMention", 0) for d in reddit_data[-24:])
        neg = sum(d.get("negativeMention", 0) for d in reddit_data[-24:])
        total_sentiment = pos + neg

        lines.append("### Reddit (last 24h)")
        lines.append(f"- **Total mentions:** {total_mentions}")
        if total_sentiment > 0:
            bull_pct = pos / total_sentiment * 100
            lines.append(f"- **Positive:** {pos} ({bull_pct:.0f}%)")
            lines.append(f"- **Negative:** {neg} ({100-bull_pct:.0f}%)")
            if bull_pct >= 65:
                lines.append("- **Signal:** Bullish retail sentiment")
            elif bull_pct <= 35:
                lines.append("- **Signal:** Bearish retail sentiment")
            else:
                lines.append("- **Signal:** Mixed sentiment")
        lines.append("")

    # Process Twitter data
    if twitter_data:
        total_mentions = sum(d.get("mention", 0) for d in twitter_data[-24:])
        pos = sum(d.get("positiveMention", 0) for d in twitter_data[-24:])
        neg = sum(d.get("negativeMention", 0) for d in twitter_data[-24:])
        total_sentiment = pos + neg

        lines.append("### Twitter/X (last 24h)")
        lines.append(f"- **Total mentions:** {total_mentions}")
        if total_sentiment > 0:
            bull_pct = pos / total_sentiment * 100
            lines.append(f"- **Positive:** {pos} ({bull_pct:.0f}%)")
            lines.append(f"- **Negative:** {neg} ({100-bull_pct:.0f}%)")
        lines.append("")

    lines.append(
        "_Social sentiment is a contrarian/confirmation signal. "
        "Extreme readings (>70% one direction) may indicate crowded trades._"
    )
    return "\n".join(lines)
