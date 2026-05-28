"""
CRO (Chief Risk Officer) Adversarial Reviewer.

Sits after Risk Judge in the graph. Reviews the recommendation by scoring
6 risk dimensions (1-5 each). Rejects only when the aggregate risk score
exceeds a configurable threshold, preventing blanket blocking.

Adapted from ATLAS-GIC Layer 4 CRO concept.
"""

import json
import re

from langchain_core.messages import AIMessage

from fund.prompts import load_prompt_or_default

_DEFAULT_PROMPT = """You are the Chief Risk Officer (CRO) conducting a final review of a trade recommendation for {company_name} on {trade_date}.

YOUR ROLE: Evaluate both the merits AND risks of this recommendation. You are a balanced risk assessor, not a blanket blocker. A good trade with manageable risks should be APPROVED.

RECOMMENDATION TO REVIEW:
{final_decision}

CONTEXT:
Market Analysis (excerpt): {market_report}
Fundamentals (excerpt): {fundamentals_report}
News (excerpt): {news_report}
Sentiment (excerpt): {sentiment_report}

SCORE EACH RISK DIMENSION from 1 (minimal risk) to 5 (critical risk):

1. MACRO HEADWINDS: Could macro factors (rates, inflation, geopolitics) overwhelm the thesis?
2. VALUATION RISK: Is the price already reflecting the upside? Is there margin of safety?
3. TECHNICAL TIMING: Do technicals suggest poor entry timing?
4. EVENT RISK: Are there imminent catalysts (earnings, FOMC, CPI) that could invalidate the thesis?
5. SENTIMENT RISK: Is the trade too crowded or sentiment at extremes?
6. THESIS QUALITY: How weak is the underlying investment thesis? (1=strong thesis, 5=no clear thesis)

IMPORTANT GUIDELINES:
- Score 1-2: Normal market conditions, thesis is solid on this dimension
- Score 3: Notable concern but not disqualifying
- Score 4: Serious concern that materially weakens the thesis
- Score 5: Critical flaw that alone could invalidate the trade
- Most dimensions for a reasonable trade should score 1-3
- Only score 4-5 when you have SPECIFIC evidence from the context, not generic concerns

After your analysis, you MUST output a JSON block with your scores:
```json
{{"macro": N, "valuation": N, "technical": N, "event": N, "sentiment": N, "thesis": N}}
```

Then state your overall assessment and: CONVICTION: [0-100]"""


# Regex to find the JSON scores block in LLM response
_SCORES_RE = re.compile(
    r'\{\s*"macro"\s*:\s*(\d)\s*,\s*"valuation"\s*:\s*(\d)\s*,\s*"technical"\s*:\s*(\d)\s*,'
    r'\s*"event"\s*:\s*(\d)\s*,\s*"sentiment"\s*:\s*(\d)\s*,\s*"thesis"\s*:\s*(\d)\s*\}'
)

# Fallback: try to find individual scores if JSON block isn't clean
_INDIVIDUAL_SCORE_RE = re.compile(r'"(\w+)"\s*:\s*(\d)')


def _parse_scores(review: str) -> dict:
    """Extract risk scores from CRO review text. Returns dict with 6 scores."""
    m = _SCORES_RE.search(review)
    if m:
        return {
            "macro": int(m.group(1)),
            "valuation": int(m.group(2)),
            "technical": int(m.group(3)),
            "event": int(m.group(4)),
            "sentiment": int(m.group(5)),
            "thesis": int(m.group(6)),
        }

    # Fallback: try to parse individual key-value pairs
    keys = {"macro", "valuation", "technical", "event", "sentiment", "thesis"}
    scores = {}
    for match in _INDIVIDUAL_SCORE_RE.finditer(review):
        key, val = match.group(1), match.group(2)
        if key in keys:
            scores[key] = min(5, max(1, int(val)))
    if len(scores) == 6:
        return scores

    # Last resort: couldn't parse scores, return moderate defaults
    return {k: 3 for k in keys}


def _parse_conviction(review: str) -> int:
    """Extract conviction score from CRO review."""
    m = re.search(r'CONVICTION\s*:\s*(\d{1,3})', review, re.IGNORECASE)
    if m:
        return min(100, max(0, int(m.group(1))))
    return 50  # default


def create_cro_reviewer(llm, prompts_dir=None, rejection_threshold=24):
    _template = load_prompt_or_default("cro_reviewer", _DEFAULT_PROMPT, prompts_dir)

    def cro_reviewer_node(state) -> dict:
        company_name = state["company_of_interest"]
        trade_date = state["trade_date"]
        final_decision = state["final_trade_decision"]
        market_report = state.get("market_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        news_report = state.get("news_report", "")
        sentiment_report = state.get("sentiment_report", "")

        print(f"[DIAG] CRO | START | ticker={company_name} | date={trade_date} "
              f"| decision_len={len(final_decision)} | threshold={rejection_threshold}")

        prompt = _template.format(
            company_name=company_name,
            trade_date=trade_date,
            final_decision=final_decision[:3000],
            market_report=market_report[:1500],
            fundamentals_report=fundamentals_report[:1500],
            news_report=news_report[:800],
            sentiment_report=sentiment_report[:800],
        )

        response = llm.invoke(prompt)
        review = response.content

        # Parse quantitative scores
        scores = _parse_scores(review)
        total_score = sum(scores.values())
        conviction = _parse_conviction(review)

        is_rejected = total_score >= rejection_threshold

        scores_str = " ".join(f"{k}={v}" for k, v in scores.items())
        print(f"[DIAG] CRO | SCORES | {scores_str} | total={total_score}/{rejection_threshold} "
              f"| conviction={conviction}")

        if is_rejected:
            new_final = (
                f"CRO OVERRIDE TO HOLD.\n\n"
                f"Risk scores: {scores} (total={total_score}, threshold={rejection_threshold})\n"
                f"Conviction: {conviction}\n\n"
                f"Original recommendation:\n{final_decision}\n\n"
                f"CRO Review:\n{review}"
            )
            print(f"[DIAG] CRO | DONE | verdict=REJECTED | total={total_score}>={rejection_threshold}")
        else:
            new_final = (
                f"{final_decision}\n\n"
                f"--- CRO REVIEW (APPROVED) ---\n"
                f"Risk scores: {scores} (total={total_score}, threshold={rejection_threshold})\n"
                f"Conviction: {conviction}\n\n"
                f"{review}"
            )
            print(f"[DIAG] CRO | DONE | verdict=APPROVED | total={total_score}<{rejection_threshold}")

        return {
            "cro_review": review,
            "final_trade_decision": new_final,
        }

    return cro_reviewer_node
