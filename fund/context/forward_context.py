"""
Forward Context Injection for Investment Fund.

Generates a text block of upcoming economic events and earnings dates
for injection into analyst prompts, giving them forward-looking awareness.

Adapted from ATLAS-GIC MiroFish seed generator's catalyst awareness.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional


# 2025-2026 FOMC meeting dates (static, updated yearly)
FOMC_DATES = [
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]


def _generate_cpi_dates(year: int) -> List[str]:
    """CPI is typically released on the 2nd or 3rd Tuesday of each month."""
    dates = []
    for month in range(1, 13):
        # Find second Tuesday
        first_day = datetime(year, month, 1)
        # Find first Tuesday
        days_until_tuesday = (1 - first_day.weekday()) % 7
        first_tuesday = first_day + timedelta(days=days_until_tuesday)
        second_tuesday = first_tuesday + timedelta(days=7)
        # CPI is often around the 12th-15th; use 2nd Tuesday as approximation
        dates.append(second_tuesday.strftime("%Y-%m-%d"))
    return dates


def _generate_nfp_dates(year: int) -> List[str]:
    """NFP is typically the first Friday of each month."""
    dates = []
    for month in range(1, 13):
        first_day = datetime(year, month, 1)
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        dates.append(first_friday.strftime("%Y-%m-%d"))
    return dates


# Pre-generate CPI and NFP dates for 2025-2026
CPI_DATES = _generate_cpi_dates(2025) + _generate_cpi_dates(2026)
NFP_DATES = _generate_nfp_dates(2025) + _generate_nfp_dates(2026)


class ForwardContextProvider:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.lookahead_days = self.config.get("forward_context_lookahead_days", 30)

    def get_forward_context(self, trade_date: str, ticker: str) -> str:
        """Generate forward context text block for injection into analyst prompts."""
        try:
            current = datetime.strptime(trade_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            return ""

        events = []

        # FOMC meetings
        for d in FOMC_DATES:
            event_date = datetime.strptime(d, "%Y-%m-%d")
            delta = (event_date - current).days
            if 0 <= delta <= self.lookahead_days:
                events.append({
                    "impact": "HIGH",
                    "event": "FOMC Meeting",
                    "date": d,
                    "days_away": delta,
                    "description": "Fed rate decision and press conference",
                })

        # CPI releases
        for d in CPI_DATES:
            event_date = datetime.strptime(d, "%Y-%m-%d")
            delta = (event_date - current).days
            if 0 <= delta <= self.lookahead_days:
                events.append({
                    "impact": "HIGH",
                    "event": "CPI Release",
                    "date": d,
                    "days_away": delta,
                    "description": "Consumer Price Index inflation data",
                })

        # NFP releases
        for d in NFP_DATES:
            event_date = datetime.strptime(d, "%Y-%m-%d")
            delta = (event_date - current).days
            if 0 <= delta <= self.lookahead_days:
                events.append({
                    "impact": "HIGH",
                    "event": "Nonfarm Payrolls",
                    "date": d,
                    "days_away": delta,
                    "description": "Monthly jobs report",
                })

        # Try to get earnings date from yfinance
        earnings_info = self._get_earnings_date(ticker, current)
        if earnings_info:
            events.append(earnings_info)

        if not events:
            return ""

        # Sort by proximity
        events.sort(key=lambda e: e["days_away"])

        # Format output
        lines = ["=== UPCOMING CATALYSTS (Next {} Days) ===".format(self.lookahead_days)]
        for e in events[:7]:  # max 7 events
            lines.append(
                f"- [{e['impact']}] {e['event']}: {e['date']} "
                f"({e['days_away']} days away) - {e['description']}"
            )
        lines.append("Consider event proximity when assessing risk/reward within the holding period.")
        lines.append("===")

        return "\n".join(lines)

    @staticmethod
    def _get_earnings_date(ticker: str, current_date: datetime) -> Optional[dict]:
        """Try to get next earnings date from yfinance."""
        try:
            import yfinance as yf
            info = yf.Ticker(ticker)
            # yfinance calendar property varies by version
            cal = getattr(info, "calendar", None)
            if cal is not None:
                # calendar can be a dict or DataFrame
                if hasattr(cal, "get"):
                    earnings_date = cal.get("Earnings Date")
                    if earnings_date and len(earnings_date) > 0:
                        ed = earnings_date[0]
                        if hasattr(ed, "strftime"):
                            delta = (ed - current_date).days
                            if 0 <= delta <= 60:
                                return {
                                    "impact": "HIGH",
                                    "event": f"{ticker} Earnings",
                                    "date": ed.strftime("%Y-%m-%d"),
                                    "days_away": delta,
                                    "description": f"Quarterly earnings report for {ticker}",
                                }
        except Exception:
            pass
        return None
