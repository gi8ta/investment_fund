from langchain_core.messages import AIMessage
from datetime import datetime, timedelta
from fund.dataflows.interface import route_to_vendor


def create_news_analyst(llm, forward_context_provider=None, config=None):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        print(f"[DIAG] NewsAnalyst | START | ticker={ticker} | date={current_date}")

        # Configurable lookback window
        cfg = config or {}
        lookback = cfg.get("news_lookback_days", 7)

        # Compute lookback start date
        start_date = (datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=lookback)).strftime("%Y-%m-%d")

        # ── Pre-fetch data ──────────────────────────────────────────
        data_sections = []

        # 1. Company-specific news
        try:
            news_data = route_to_vendor("get_news", ticker, start_date, current_date)
            data_sections.append(f"## Company-Specific News ({start_date} to {current_date})\n{news_data}")
            print(f"[DIAG] NewsAnalyst | PREFETCH | get_news OK | len={len(str(news_data))}")
        except Exception as e:
            print(f"[DIAG] NewsAnalyst | PREFETCH | get_news FAILED | {e}")

        # 2. Global/macro news
        try:
            global_news = route_to_vendor("get_global_news", current_date, lookback, 10)
            data_sections.append(f"## Global & Macroeconomic News\n{global_news}")
            print(f"[DIAG] NewsAnalyst | PREFETCH | get_global_news OK | len={len(str(global_news))}")
        except Exception as e:
            print(f"[DIAG] NewsAnalyst | PREFETCH | get_global_news FAILED | {e}")

        # 3. Macroeconomic data snapshot (FRED)
        try:
            macro_data = route_to_vendor("get_macro_snapshot", current_date, 90)
            data_sections.append(f"## Macroeconomic Indicators\n{macro_data}")
            print(f"[DIAG] NewsAnalyst | PREFETCH | get_macro_snapshot OK | len={len(str(macro_data))}")
        except Exception as e:
            print(f"[DIAG] NewsAnalyst | PREFETCH | get_macro_snapshot FAILED | {e}")

        # 4. Forward context (upcoming catalysts) if available
        if forward_context_provider:
            try:
                fwd_ctx = forward_context_provider.get_forward_context(current_date, ticker)
                if fwd_ctx:
                    data_sections.append(fwd_ctx)
            except Exception:
                pass

        fetched_data = "\n\n".join(data_sections) if data_sections else "No news data available."

        # ── LLM analysis ────────────────────────────────────────────
        system_message = f"""You are a news researcher tasked with analyzing recent news and trends. Below is the pre-fetched news data for {ticker} and the broader market as of {current_date}.

{fetched_data}

Write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Cover both company-specific developments and broader market trends. Analyze how macro events (interest rates, geopolitics, sector rotation) might impact {ticker}.

When macroeconomic data is available, interpret the numbers: rising CPI suggests tightening, low unemployment supports consumer spending, yield curve inversion signals recession risk, high VIX suggests elevated fear.

Do not simply state the trends are mixed — provide detailed and fine-grained analysis and insights that may help traders make decisions.

Make sure to append a Markdown table at the end of the report to organize key points, organized and easy to read."""

        messages = [
            ("system", system_message),
            ("human", f"Please analyze the news data and write your comprehensive report for {ticker}."),
        ]

        result = llm.invoke(messages)
        report = result.content
        print(f"[DIAG] NewsAnalyst | DONE | report_len={len(report)}")

        return {
            "messages": [AIMessage(content=report, name="NewsAnalyst")],
            "news_report": report,
        }

    return news_analyst_node
