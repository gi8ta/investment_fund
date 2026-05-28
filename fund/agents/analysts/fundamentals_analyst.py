from langchain_core.messages import AIMessage
from fund.dataflows.interface import route_to_vendor


def create_fundamentals_analyst(llm, forward_context_provider=None, config=None):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        print(f"[DIAG] FundamentalsAnalyst | START | ticker={ticker} | date={current_date}")

        # ── Pre-fetch data ──────────────────────────────────────────
        data_sections = []

        # 1. Comprehensive fundamentals overview
        try:
            fundamentals = route_to_vendor("get_fundamentals", ticker, current_date)
            data_sections.append(f"## Company Fundamentals Overview\n{fundamentals}")
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_fundamentals OK | len={len(str(fundamentals))}")
        except Exception as e:
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_fundamentals FAILED | {e}")

        # 2. Balance sheet
        try:
            balance_sheet = route_to_vendor("get_balance_sheet", ticker, "quarterly", current_date)
            data_sections.append(f"## Balance Sheet (Quarterly)\n{balance_sheet}")
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_balance_sheet OK | len={len(str(balance_sheet))}")
        except Exception as e:
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_balance_sheet FAILED | {e}")

        # 3. Cash flow statement
        try:
            cashflow = route_to_vendor("get_cashflow", ticker, "quarterly", current_date)
            data_sections.append(f"## Cash Flow Statement (Quarterly)\n{cashflow}")
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_cashflow OK | len={len(str(cashflow))}")
        except Exception as e:
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_cashflow FAILED | {e}")

        # 4. Income statement
        try:
            income_stmt = route_to_vendor("get_income_statement", ticker, "quarterly", current_date)
            data_sections.append(f"## Income Statement (Quarterly)\n{income_stmt}")
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_income_statement OK | len={len(str(income_stmt))}")
        except Exception as e:
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_income_statement FAILED | {e}")

        # 5. Earnings surprise history
        try:
            earnings = route_to_vendor("get_earnings_surprise", ticker, current_date)
            data_sections.append(f"## Earnings Surprise History\n{earnings}")
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_earnings_surprise OK | len={len(str(earnings))}")
        except Exception as e:
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_earnings_surprise FAILED | {e}")

        # 6. Insider transactions
        try:
            insider_txns = route_to_vendor("get_insider_transactions", ticker)
            data_sections.append(f"## Insider Transactions\n{insider_txns}")
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_insider_transactions OK | len={len(str(insider_txns))}")
        except Exception as e:
            print(f"[DIAG] FundamentalsAnalyst | PREFETCH | get_insider_transactions FAILED | {e}")

        # 7. Forward context (upcoming catalysts) if available
        if forward_context_provider:
            try:
                fwd_ctx = forward_context_provider.get_forward_context(current_date, ticker)
                if fwd_ctx:
                    data_sections.append(fwd_ctx)
            except Exception:
                pass

        fetched_data = "\n\n".join(data_sections) if data_sections else "No fundamentals data available."

        # ── LLM analysis ────────────────────────────────────────────
        system_message = f"""You are a fundamental analyst tasked with analyzing company financials. Below is the pre-fetched fundamental data for {ticker} as of {current_date}.

{fetched_data}

Write a comprehensive report covering:
- Financial health (liquidity ratios, debt levels, working capital)
- Profitability (margins, ROE, ROA, earnings trends)
- Cash flow quality (operating cash flow vs net income, free cash flow)
- Growth indicators (revenue growth, earnings growth, guidance)
- Valuation context (P/E, P/B, EV/EBITDA relative to sector)
- Earnings execution (beat/miss history, trend in surprises)
- Insider activity (are insiders buying or selling? large transactions?)

Do not simply state the trends are mixed — provide detailed and fine-grained analysis and insights that may help traders make decisions.

Make sure to append a Markdown table at the end of the report to organize key points, organized and easy to read."""

        messages = [
            ("system", system_message),
            ("human", f"Please analyze the fundamental data for {ticker} and write your comprehensive report."),
        ]

        result = llm.invoke(messages)
        report = result.content
        print(f"[DIAG] FundamentalsAnalyst | DONE | report_len={len(report)}")

        return {
            "messages": [AIMessage(content=report, name="FundamentalsAnalyst")],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
