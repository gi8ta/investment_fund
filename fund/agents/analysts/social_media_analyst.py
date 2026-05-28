from langchain_core.messages import AIMessage
from datetime import datetime, timedelta
from fund.dataflows.interface import route_to_vendor


def create_social_media_analyst(llm, forward_context_provider=None, config=None):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        print(f"[DIAG] SocialMediaAnalyst | START | ticker={ticker} | date={current_date}")

        # Configurable lookback window
        cfg = config or {}
        lookback = cfg.get("news_lookback_days", 7)

        # ── Pre-fetch data ──────────────────────────────────────────
        data_sections = []

        # 1. Social sentiment (Finnhub Reddit/Twitter, with StockTwits fallback)
        try:
            sentiment_data = route_to_vendor("get_social_sentiment", ticker, current_date)
            data_sections.append(f"## Social Media Sentiment\n{sentiment_data}")
            print(f"[DIAG] SocialMediaAnalyst | PREFETCH | get_social_sentiment OK | len={len(str(sentiment_data))}")
        except Exception as e:
            print(f"[DIAG] SocialMediaAnalyst | PREFETCH | get_social_sentiment FAILED | {e}")

        # 2. Insider sentiment (MSPR — institutional signal)
        try:
            insider_data = route_to_vendor("get_insider_sentiment", ticker, current_date)
            data_sections.append(f"## Insider Sentiment (MSPR)\n{insider_data}")
            print(f"[DIAG] SocialMediaAnalyst | PREFETCH | get_insider_sentiment OK | len={len(str(insider_data))}")
        except Exception as e:
            print(f"[DIAG] SocialMediaAnalyst | PREFETCH | get_insider_sentiment FAILED | {e}")

        # 3. Analyst ratings (institutional consensus)
        try:
            ratings_data = route_to_vendor("get_analyst_ratings", ticker, current_date)
            data_sections.append(f"## Analyst Ratings\n{ratings_data}")
            print(f"[DIAG] SocialMediaAnalyst | PREFETCH | get_analyst_ratings OK | len={len(str(ratings_data))}")
        except Exception as e:
            print(f"[DIAG] SocialMediaAnalyst | PREFETCH | get_analyst_ratings FAILED | {e}")

        # 4. Forward context (upcoming catalysts) if available
        if forward_context_provider:
            try:
                fwd_ctx = forward_context_provider.get_forward_context(current_date, ticker)
                if fwd_ctx:
                    data_sections.append(fwd_ctx)
            except Exception:
                pass

        fetched_data = "\n\n".join(data_sections) if data_sections else "No social/sentiment data available."

        # ── LLM analysis ────────────────────────────────────────────
        system_message = f"""You are a social media and sentiment researcher/analyst. Below is the pre-fetched social sentiment, insider activity, and analyst consensus data for {ticker} as of {current_date}.

{fetched_data}

Write a comprehensive report detailing your analysis, insights, and implications for traders and investors.

Key analysis points:
- Social sentiment (Reddit/Twitter): Extreme readings (>70% one direction) may indicate crowded trades and potential reversals. Cross-reference with price action.
- Insider sentiment (MSPR): MSPR >0.6 = net insider buying (bullish), <0.4 = net insider selling (bearish). Insiders have informational advantage.
- Analyst consensus: Track upgrades/downgrades and the buy/hold/sell ratio. Consensus shifts often precede price moves.

Look for convergence or divergence between these signals — when retail, insiders, and analysts agree, the signal is stronger. When they diverge, identify which group is likely more informed.

Do not simply state the trends are mixed — provide detailed and fine-grained analysis and insights that may help traders make decisions.

Make sure to append a Markdown table at the end of the report to organize key points, organized and easy to read."""

        messages = [
            ("system", system_message),
            ("human", f"Please analyze the social media sentiment and insider/analyst data for {ticker} and write your comprehensive report."),
        ]

        result = llm.invoke(messages)
        report = result.content
        print(f"[DIAG] SocialMediaAnalyst | DONE | report_len={len(report)}")

        return {
            "messages": [AIMessage(content=report, name="SocialMediaAnalyst")],
            "sentiment_report": report,
        }

    return social_media_analyst_node
