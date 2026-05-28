import time
import json

from fund.prompts import load_prompt_or_default

_DEFAULT_PROMPT = """As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts--Risky, Neutral, and Safe/Conservative--and determine the best course of action for the trader. Your decision must result in a clear recommendation: Buy, Sell, or Hold. Choose Hold only if strongly justified by specific arguments, not as a fallback when all sides seem valid. Strive for clarity and decisiveness.

Guidelines for Decision-Making:
1. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context.
2. **Provide Rationale**: Support your recommendation with direct quotes and counterarguments from the debate.
3. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on the analysts' insights.
4. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments and improve the decision you are making now to make sure you don't make a wrong BUY/SELL/HOLD call that loses money.

Deliverables:
- A clear and actionable recommendation: Buy, Sell, or Hold.
- Detailed reasoning anchored in the debate and past reflections.

---
{weight_context}
**Analysts Debate History:**
{history}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""


def create_risk_manager(llm, memory, darwinian_weights=None, prompts_dir=None):
    _template = load_prompt_or_default("risk_manager", _DEFAULT_PROMPT, prompts_dir)

    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        # When --skip-risk-debate is set, the three risk debators never
        # ran, so risk_debate_state may be missing or partially populated.
        risk_debate_state = state.get("risk_debate_state", {}) or {}
        history = risk_debate_state.get("history", "")
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        weight_context = ""
        if darwinian_weights:
            weight_context = darwinian_weights.format_weight_context_for_risk_manager()

        print(f"[DIAG] RiskManager | START | ticker={company_name} | memories={len(past_memories)} | risk_rounds={risk_debate_state.get('count', 0)} | darwinian={'yes' if weight_context else 'no'}")

        prompt = _template.format(
            trader_plan=trader_plan,
            past_memory_str=past_memory_str,
            weight_context=weight_context,
            history=history,
        )

        response = llm.invoke(prompt)
        print(f"[DIAG] RiskManager | DONE | final_decision_len={len(response.content)} | snippet={response.content[:300]!r}")

        # Use .get() with empty defaults so the node also works when
        # --skip-risk-debate is on and the three risk debators never
        # populated the *_history / current_* fields.
        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state.get("history", ""),
            "risky_history": risk_debate_state.get("risky_history", ""),
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state.get("current_risky_response", ""),
            "current_safe_response": risk_debate_state.get("current_safe_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "count": risk_debate_state.get("count", 0),
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
