from langchain_core.messages import AIMessage
from datetime import datetime, timedelta
from fund.dataflows.interface import route_to_vendor


def create_market_analyst(llm, forward_context_provider=None, config=None):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        print(f"[DIAG] MarketAnalyst | START | ticker={ticker} | date={current_date}")

        # Configurable lookback window
        cfg = config or {}
        lookback = cfg.get("market_lookback_days", 30)

        # Compute lookback start date
        start_date = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=lookback)).strftime("%Y-%m-%d")

        # ── Pre-fetch data ──────────────────────────────────────────
        data_sections = []

        # 1. Stock price data (OHLCV)
        try:
            stock_data = route_to_vendor("get_stock_data", ticker, start_date, current_date)
            data_sections.append(f"## Stock Price Data (OHLCV, {start_date} to {current_date})\n{stock_data}")
            print(f"[DIAG] MarketAnalyst | PREFETCH | get_stock_data OK | len={len(str(stock_data))}")
        except Exception as e:
            print(f"[DIAG] MarketAnalyst | PREFETCH | get_stock_data FAILED | {e}")

        # 2. Technical indicators (expanded set)
        indicators = [
            "rsi", "macd", "macds", "macdh",
            "boll", "boll_ub", "boll_lb",
            "atr", "mfi", "vwma",
        ]
        for indicator in indicators:
            try:
                ind_data = route_to_vendor("get_indicators", ticker, indicator, current_date, lookback)
                data_sections.append(f"## Indicator: {indicator}\n{ind_data}")
                print(f"[DIAG] MarketAnalyst | PREFETCH | get_indicators({indicator}) OK")
            except Exception as e:
                print(f"[DIAG] MarketAnalyst | PREFETCH | get_indicators({indicator}) FAILED | {e}")

        # 3. VIX (market fear/greed proxy)
        try:
            vix_data = route_to_vendor("get_vix", current_date, lookback)
            data_sections.append(f"## Market Volatility\n{vix_data}")
            print(f"[DIAG] MarketAnalyst | PREFETCH | get_vix OK")
        except Exception as e:
            print(f"[DIAG] MarketAnalyst | PREFETCH | get_vix FAILED | {e}")

        # 4. Forward context (upcoming catalysts) if available
        if forward_context_provider:
            try:
                fwd_ctx = forward_context_provider.get_forward_context(current_date, ticker)
                if fwd_ctx:
                    data_sections.append(fwd_ctx)
                    print(f"[DIAG] MarketAnalyst | PREFETCH | forward_context OK | len={len(fwd_ctx)}")
            except Exception as e:
                print(f"[DIAG] MarketAnalyst | PREFETCH | forward_context FAILED | {e}")

        fetched_data = "\n\n".join(data_sections) if data_sections else "No market data available."

        # ── LLM analysis ────────────────────────────────────────────
        system_message = f"""You are a trading assistant tasked with analyzing financial markets. Below is the pre-fetched market data for {ticker} as of {current_date}.

{fetched_data}

Analyze the data above and write a detailed, nuanced report of the trends you observe. The indicators provided are:
- RSI: Momentum — overbought >70, oversold <30
- MACD / Signal / Histogram: Trend momentum and crossovers
- Bollinger Bands (middle, upper, lower): Volatility and mean reversion
- ATR: Volatility measurement for risk sizing
- MFI: Money Flow Index — volume-weighted RSI, overbought >80, oversold <20
- VWMA: Volume-Weighted Moving Average — confirms trends with volume
- VIX: Market fear gauge — >30 extreme fear, <15 complacency

Do not simply state the trends are mixed — provide detailed and fine-grained analysis and insights that may help traders make decisions. Include analysis of price trends, momentum, volatility patterns, and support/resistance levels.

Make sure to append a Markdown table at the end of the report to organize key points, organized and easy to read."""

        messages = [
            ("system", system_message),
            ("human", f"Please analyze the market data for {ticker} and write your comprehensive report."),
        ]

        result = llm.invoke(messages)
        report = result.content
        print(f"[DIAG] MarketAnalyst | DONE | report_len={len(report)}")

        return {
            "messages": [AIMessage(content=report, name="MarketAnalyst")],
            "market_report": report,
        }

    return market_analyst_node
