"""
Per-agent Sharpe ratio computation for Autoresearch.

Maintains a rolling window of trade outcomes per agent and computes
risk-adjusted performance (Sharpe ratio) to identify the weakest agent
whose prompt should be modified next.

Adapted from ATLAS-GIC autoresearch scoring.
"""

import math
from typing import Dict, List, Optional, Tuple


class AgentScorecard:
    """Computes rolling Sharpe ratios for each debating agent."""

    # Agents whose prompts can be modified by autoresearch
    MODIFIABLE_AGENTS = [
        "bull_researcher",
        "bear_researcher",
        "research_manager",
        "risk_manager",
        "trader",
        "risky_debator",
        "safe_debator",
        "neutral_debator",
    ]

    def __init__(self, window: int = 60):
        self.window = window
        # agent_name -> list of (weighted_return, conviction)
        self._history: Dict[str, List[dict]] = {a: [] for a in self.MODIFIABLE_AGENTS}

    def record(
        self,
        agent: str,
        actual_return_pct: float,
        conviction: float = 50.0,
        direction_sign: float = 1.0,
    ) -> None:
        """Record a trade outcome for an agent.

        Args:
            agent: Agent name (must be in MODIFIABLE_AGENTS)
            actual_return_pct: Signed return for BUY direction
            conviction: Conviction 0-100 from the agent's output
            direction_sign: +1 for BUY recommendation, -1 for SELL, 0 for HOLD
        """
        if agent not in self._history:
            return
        weighted_ret = actual_return_pct * (conviction / 100.0) * direction_sign
        self._history[agent].append({
            "weighted_return": weighted_ret,
            "actual_return": actual_return_pct,
            "conviction": conviction,
            "direction_sign": direction_sign,
        })
        # Trim to window
        if len(self._history[agent]) > self.window:
            self._history[agent] = self._history[agent][-self.window:]

    def compute_sharpe(self, agent: str) -> Optional[float]:
        """Compute annualized Sharpe ratio for an agent.

        Returns None if fewer than 5 data points.
        """
        records = self._history.get(agent, [])
        if len(records) < 5:
            return None
        returns = [r["weighted_return"] for r in records]
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
        std = math.sqrt(var) if var > 0 else 1e-8
        return (mean_r / std) * math.sqrt(252)

    def compute_all_sharpes(self) -> Dict[str, Optional[float]]:
        """Compute Sharpe ratios for all modifiable agents."""
        return {a: self.compute_sharpe(a) for a in self.MODIFIABLE_AGENTS}

    def find_weakest_agent(self) -> Optional[Tuple[str, float]]:
        """Find the agent with the lowest Sharpe ratio.

        Returns (agent_name, sharpe) or None if insufficient data.
        """
        sharpes = self.compute_all_sharpes()
        valid = {a: s for a, s in sharpes.items() if s is not None}
        if not valid:
            return None
        worst = min(valid, key=lambda a: valid[a])
        return (worst, valid[worst])

    def get_history_summary(self, agent: str, last_n: int = 10) -> str:
        """Get a text summary of recent trade performance for an agent."""
        records = self._history.get(agent, [])[-last_n:]
        if not records:
            return f"No trade history for {agent}."
        lines = [f"Recent {len(records)} trades for {agent}:"]
        for i, r in enumerate(records, 1):
            lines.append(
                f"  {i}. return={r['actual_return']:+.2f}% "
                f"conviction={r['conviction']:.0f} "
                f"dir={'+' if r['direction_sign'] > 0 else '-' if r['direction_sign'] < 0 else '0'} "
                f"weighted={r['weighted_return']:+.4f}"
            )
        sharpe = self.compute_sharpe(agent)
        lines.append(f"  Sharpe: {sharpe:.3f}" if sharpe is not None else "  Sharpe: N/A")
        return "\n".join(lines)
