import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("FUND_RESULTS_DIR", "./results"),
    "data_dir": os.getenv("FUND_DATA_DIR", "./data"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://api.openai.com/v1",
    # Debate and discussion settings
    "max_debate_rounds": 3,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: yfinance, alpha_vantage, local
        "technical_indicators": "yfinance",  # Options: yfinance, alpha_vantage, local
        "fundamental_data": "yfinance",        # Options: yfinance, alpha_vantage, finnhub, local
        "news_data": "finnhub",                # Options: finnhub, alpha_vantage, google, local
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
        # Example: "get_news": "openai",               # Override category default
    },
    # Analyst lookback windows (configurable per trading horizon)
    "market_lookback_days": 30,     # Market Analyst: OHLCV + indicators window
    "news_lookback_days": 7,        # News/Social Analyst: news search window
    # ATLAS-GIC integrations (all disabled by default)
    "enable_darwinian_weights": False,
    "darwinian_weights_file": "darwinian_weights.json",
    "enable_cro": False,
    "cro_rejection_threshold": 24,  # Total risk score (6 dims x 1-5) needed to reject. Max=30, default=24 (~80%)
    "enable_forward_context": False,
    "forward_context_lookahead_days": 30,
    "enable_autoresearch": False,
    "autoresearch_trigger_every": 20,
    "autoresearch_sharpe_threshold": 0.0,
    "prompts_dir": "fund/prompts",
    "enable_janus": False,
    "janus_cohorts": [],
    "janus_state_file": "janus_state.json",
}
