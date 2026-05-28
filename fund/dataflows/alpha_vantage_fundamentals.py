import json
from .alpha_vantage_common import _make_api_request

# Maximum number of quarterly/annual reports to keep
MAX_QUARTERLY_REPORTS = 4
MAX_ANNUAL_REPORTS = 1


def _truncate_financial_reports(raw_response: str) -> str:
    """Truncate financial statement JSON to recent reports only.

    Alpha Vantage returns 20+ quarters of data.  We keep only the most
    recent MAX_QUARTERLY_REPORTS quarterly and MAX_ANNUAL_REPORTS annual
    reports to avoid blowing up the LLM context window.
    """
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        # Not JSON (CSV or error) — return as-is
        return raw_response

    if not isinstance(data, dict):
        return raw_response

    changed = False
    if "quarterlyReports" in data and len(data["quarterlyReports"]) > MAX_QUARTERLY_REPORTS:
        data["quarterlyReports"] = data["quarterlyReports"][:MAX_QUARTERLY_REPORTS]
        changed = True
    if "annualReports" in data and len(data["annualReports"]) > MAX_ANNUAL_REPORTS:
        data["annualReports"] = data["annualReports"][:MAX_ANNUAL_REPORTS]
        changed = True

    if changed:
        return json.dumps(data, indent=2)
    return raw_response


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol using Alpha Vantage.
    Limited to the most recent quarters to keep context manageable.
    """
    params = {
        "symbol": ticker,
    }

    raw = _make_api_request("BALANCE_SHEET", params)
    return _truncate_financial_reports(raw)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage.
    Limited to the most recent quarters to keep context manageable.
    """
    params = {
        "symbol": ticker,
    }

    raw = _make_api_request("CASH_FLOW", params)
    return _truncate_financial_reports(raw)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None) -> str:
    """
    Retrieve income statement data for a given ticker symbol using Alpha Vantage.
    Limited to the most recent quarters to keep context manageable.
    """
    params = {
        "symbol": ticker,
    }

    raw = _make_api_request("INCOME_STATEMENT", params)
    return _truncate_financial_reports(raw)
