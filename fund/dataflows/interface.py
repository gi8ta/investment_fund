from typing import Annotated

# Import from vendor-specific modules
from .y_finance import get_YFin_data_online, get_stock_stats_indicators_window, get_balance_sheet as get_yfinance_balance_sheet, get_cashflow as get_yfinance_cashflow, get_income_statement as get_yfinance_income_statement, get_insider_transactions as get_yfinance_insider_transactions
from .yfinance_fundamentals import get_fundamentals_overview as get_yfinance_fundamentals
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news
)
from .alpha_vantage_common import AlphaVantageRateLimitError
from .finnhub_live import (
    get_company_news as get_finnhub_live_news,
    get_market_news as get_finnhub_market_news,
    get_insider_sentiment as get_finnhub_live_insider_sentiment,
    get_basic_financials as get_finnhub_basic_financials,
    get_earnings_surprise as get_finnhub_earnings_surprise,
    get_analyst_ratings as get_finnhub_analyst_ratings,
    get_social_sentiment as get_finnhub_social_sentiment,
)
try:
    from .stocktwits import get_ticker_sentiment as get_stocktwits_sentiment
except ImportError:
    get_stocktwits_sentiment = None
from .fred_macro import get_macro_snapshot as get_fred_macro_snapshot, get_vix_data as get_fred_vix_data
from .api_cache import cache_get, cache_set, _make_key

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement",
            "get_earnings_surprise",
            "get_analyst_ratings",
        ]
    },
    "news_data": {
        "description": "News (public/insiders, original/processed, social sentiment)",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_sentiment",
            "get_insider_transactions",
            "get_social_sentiment",
            "get_macro_snapshot",
            "get_vix",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "finnhub",
    "stocktwits",
    "fred",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance":      get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance":      get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "yfinance":      get_yfinance_fundamentals,         # primary: point-in-time safe
        "alpha_vantage": get_alpha_vantage_fundamentals,     # fallback: comprehensive overview
        "finnhub":       get_finnhub_basic_financials,       # fallback: P/E, EPS, beta, ratios
    },
    "get_balance_sheet": {
        "yfinance":      get_yfinance_balance_sheet,         # primary: filtered by curr_date
        "alpha_vantage": get_alpha_vantage_balance_sheet,
    },
    "get_cashflow": {
        "yfinance":      get_yfinance_cashflow,              # primary: filtered by curr_date
        "alpha_vantage": get_alpha_vantage_cashflow,
    },
    "get_income_statement": {
        "yfinance":      get_yfinance_income_statement,      # primary: filtered by curr_date
        "alpha_vantage": get_alpha_vantage_income_statement,
    },
    "get_earnings_surprise": {
        "finnhub": get_finnhub_earnings_surprise,            # EPS actual vs estimate
    },
    "get_analyst_ratings": {
        "finnhub": get_finnhub_analyst_ratings,              # buy/hold/sell consensus
    },
    # news_data
    "get_news": {
        "finnhub":       get_finnhub_live_news,              # primary: 60 req/min, date-range
        "alpha_vantage": get_alpha_vantage_news,             # fallback: includes sentiment scores
    },
    "get_global_news": {
        "finnhub": get_finnhub_market_news,                  # general market/macro news
    },
    "get_insider_sentiment": {
        "finnhub": get_finnhub_live_insider_sentiment,       # MSPR data
    },
    "get_insider_transactions": {
        "yfinance":      get_yfinance_insider_transactions,  # primary: filtered by curr_date
        "alpha_vantage": get_alpha_vantage_insider_transactions,
    },
    "get_social_sentiment": {
        "finnhub":    get_finnhub_social_sentiment,          # primary: Reddit + Twitter
        "stocktwits": get_stocktwits_sentiment,              # fallback: requires token
    },
    "get_macro_snapshot": {
        "fred": get_fred_macro_snapshot,                     # CPI, rates, unemployment, VIX
    },
    "get_vix": {
        "fred": get_fred_vix_data,                           # VIX with yfinance fallback built-in
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support.

    Features:
    - SQLite cache layer (avoids redundant API calls)
    - Automatic fallback through all available vendors
    - Graceful degradation (returns informational string instead of crashing)
    """
    category = get_category_for_method(method)

    # ── Cache check ──────────────────────────────────────────
    cache_key = _make_key(method, args, kwargs)
    cached = cache_get(cache_key, category)
    if cached is not None:
        print(f"CACHE_HIT: {method} (key={cache_key[:8]}...)")
        return cached

    # ── Vendor routing ───────────────────────────────────────
    vendor_config = get_vendor(category, method)

    # Handle comma-separated vendors (defines fallback order)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Get all available vendors for this method for fallback
    all_available_vendors = list(VENDOR_METHODS[method].keys())

    # Create fallback vendor list: primary vendors first, then remaining vendors as fallbacks
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    # Debug: Print fallback ordering
    fallback_str = " → ".join(fallback_vendors)
    print(f"DEBUG: {method} | fallback order: [{fallback_str}]")

    vendor_attempt_count = 0

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            if vendor in primary_vendors:
                print(f"INFO: Vendor '{vendor}' not supported for method '{method}', skipping")
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        vendor_attempt_count += 1

        vendor_type = "PRIMARY" if vendor in primary_vendors else "FALLBACK"
        print(f"DEBUG: Attempting {vendor_type} vendor '{vendor}' for {method} (attempt #{vendor_attempt_count})")

        # Handle list of methods for a vendor
        if isinstance(vendor_impl, list):
            vendor_methods = [(impl, vendor) for impl in vendor_impl]
        else:
            vendor_methods = [(vendor_impl, vendor)]

        # Run methods for this vendor
        vendor_results = []
        for impl_func, vendor_name in vendor_methods:
            try:
                print(f"DEBUG: Calling {impl_func.__name__} from vendor '{vendor_name}'...")
                result = impl_func(*args, **kwargs)
                vendor_results.append(result)
                print(f"SUCCESS: {impl_func.__name__} from vendor '{vendor_name}' completed successfully")

            except AlphaVantageRateLimitError as e:
                if vendor == "alpha_vantage":
                    print(f"RATE_LIMIT: Alpha Vantage rate limit exceeded, falling back to next vendor")
                    print(f"DEBUG: Rate limit details: {e}")
                continue
            except Exception as e:
                print(f"FAILED: {impl_func.__name__} from vendor '{vendor_name}' failed: {e}")
                continue

        # Stop after first successful vendor (single-vendor mode)
        if vendor_results:
            print(f"SUCCESS: Vendor '{vendor}' succeeded with {len(vendor_results)} result(s)")
            result = vendor_results[0] if len(vendor_results) == 1 else '\n'.join(str(r) for r in vendor_results)

            # ── Cache store ──────────────────────────────────
            try:
                cache_set(cache_key, str(result), category)
            except Exception:
                pass  # Don't fail on cache errors

            return result
        else:
            print(f"FAILED: Vendor '{vendor}' produced no results")

    # All vendors failed — graceful degradation instead of crash
    print(f"WARNING: All {vendor_attempt_count} vendor attempts failed for method '{method}' — returning empty data")
    return f"Data unavailable: all vendors failed for {method}. Analysis should proceed with available information."
