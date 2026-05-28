# fund/graph/conditional_logic.py

from fund.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            print(f"[DIAG] Routing | MarketAnalyst → tools_market | tool_calls={len(last_message.tool_calls)}")
            return "tools_market"
        print(f"[DIAG] Routing | MarketAnalyst → Msg Clear Market (report ready)")
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            print(f"[DIAG] Routing | SocialAnalyst → tools_social | tool_calls={len(last_message.tool_calls)}")
            return "tools_social"
        print(f"[DIAG] Routing | SocialAnalyst → Msg Clear Social (report ready)")
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            print(f"[DIAG] Routing | NewsAnalyst → tools_news | tool_calls={len(last_message.tool_calls)}")
            return "tools_news"
        print(f"[DIAG] Routing | NewsAnalyst → Msg Clear News (report ready)")
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            print(f"[DIAG] Routing | FundamentalsAnalyst → tools_fundamentals | tool_calls={len(last_message.tool_calls)}")
            return "tools_fundamentals"
        print(f"[DIAG] Routing | FundamentalsAnalyst → Msg Clear Fundamentals (report ready)")
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""
        count = state["investment_debate_state"]["count"]
        max_count = 2 * self.max_debate_rounds
        if count >= max_count:
            print(f"[DIAG] Routing | Debate → Research Manager | count={count} >= max={max_count}")
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            print(f"[DIAG] Routing | Debate → Bear Researcher | count={count}/{max_count}")
            return "Bear Researcher"
        print(f"[DIAG] Routing | Debate → Bull Researcher | count={count}/{max_count}")
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        count = state["risk_debate_state"]["count"]
        max_count = 3 * self.max_risk_discuss_rounds
        if count >= max_count:
            print(f"[DIAG] Routing | RiskDebate → Risk Judge | count={count} >= max={max_count}")
            return "Risk Judge"
        speaker = state["risk_debate_state"]["latest_speaker"]
        if speaker.startswith("Risky"):
            print(f"[DIAG] Routing | RiskDebate → Safe Analyst | count={count}/{max_count}")
            return "Safe Analyst"
        if speaker.startswith("Safe"):
            print(f"[DIAG] Routing | RiskDebate → Neutral Analyst | count={count}/{max_count}")
            return "Neutral Analyst"
        print(f"[DIAG] Routing | RiskDebate → Risky Analyst | count={count}/{max_count}")
        return "Risky Analyst"
