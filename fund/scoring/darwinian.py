"""
Darwinian weight tracking and updating for Investment Fund.

Tracks per-agent accuracy over time. After each trade, scores each debating
agent's directional call against the actual outcome. Periodically applies
quartile-based multiplicative weight updates (top quartile boosted, bottom
quartile decayed). Weights are persisted to a JSON file.

Adapted from ATLAS-GIC's Darwinian weighting system.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


AGENTS = [
    "bull_researcher",
    "bear_researcher",
    "risky_analyst",
    "safe_analyst",
    "neutral_analyst",
]

DEFAULT_WEIGHT = 1.0
MIN_WEIGHT = 0.3
MAX_WEIGHT = 2.5
BOOST_FACTOR = 1.05
DECAY_FACTOR = 0.95
MIN_TRADES_FOR_QUARTILE = 4


class DarwinianWeights:
    def __init__(self, state_file: str = "darwinian_weights.json"):
        self.state_file = Path(state_file)
        self.weights: Dict[str, float] = {a: DEFAULT_WEIGHT for a in AGENTS}
        self.trade_history: List[dict] = []
        self.total_updates: int = 0
        self._load_state()

    # ── persistence ──────────────────────────────────────────

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            self.weights = data.get("weights", self.weights)
            self.trade_history = data.get("trade_history", [])
            self.total_updates = data.get("total_updates", 0)
        except (json.JSONDecodeError, IOError):
            pass

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({
            "weights": self.weights,
            "trade_history": self.trade_history[-500:],  # keep last 500
            "total_updates": self.total_updates,
        }, indent=2))

    # ── public API ───────────────────────────────────────────

    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def get_weight(self, agent: str) -> float:
        return self.weights.get(agent, DEFAULT_WEIGHT)

    def record_trade_outcome(
        self,
        ticker: str,
        trade_date: str,
        signal: str,
        actual_return_pct: float,
        state: dict,
    ) -> Dict[str, float]:
        """Score each agent against the actual outcome and update weights.

        Args:
            ticker: Stock ticker
            trade_date: Signal date
            signal: Final system signal (BUY/SELL/HOLD)
            actual_return_pct: Signed return for BUY direction
            state: Full AgentState dict from the propagation

        Returns:
            Updated weights dict.
        """
        scores: Dict[str, float] = {}

        # Score bull_researcher: correct when market went up
        scores["bull_researcher"] = 1.0 if actual_return_pct > 0 else 0.0

        # Score bear_researcher: correct when market went down
        scores["bear_researcher"] = 1.0 if actual_return_pct < 0 else 0.0

        # Score risk debators based on their directional lean
        risk_state = state.get("risk_debate_state", {})
        for agent_key, history_key in [
            ("risky_analyst", "current_risky_response"),
            ("safe_analyst", "current_safe_response"),
            ("neutral_analyst", "current_neutral_response"),
        ]:
            response = risk_state.get(history_key, "")
            direction = self._extract_direction(response)
            if direction == "BUY":
                scores[agent_key] = 1.0 if actual_return_pct > 0 else 0.0
            elif direction == "SELL":
                scores[agent_key] = 1.0 if actual_return_pct < 0 else 0.0
            else:
                scores[agent_key] = 0.5  # HOLD / unknown → neutral score

        # Record history
        for agent, score in scores.items():
            self.trade_history.append({
                "agent": agent,
                "ticker": ticker,
                "trade_date": trade_date,
                "signal": signal,
                "actual_return_pct": round(actual_return_pct, 4),
                "score": score,
            })

        # Apply quartile update if enough trades
        self._update_weights_quartile()
        self.total_updates += 1
        self._save_state()

        print(f"[DIAG] Darwinian | trade={ticker} {trade_date} | ret={actual_return_pct:+.2f}% | scores={scores} | weights={self.weights}")
        return dict(self.weights)

    # ── internal ─────────────────────────────────────────────

    @staticmethod
    def _extract_direction(text: str) -> str:
        """Extract directional lean (BUY/SELL/HOLD) from a debator response."""
        if not text:
            return "HOLD"
        upper = text.upper()
        # Look for explicit recommendation keywords
        buy_patterns = [r'\bBUY\b', r'\bLONG\b', r'\bBULLISH\b']
        sell_patterns = [r'\bSELL\b', r'\bSHORT\b', r'\bBEARISH\b']
        buy_count = sum(1 for p in buy_patterns if re.search(p, upper))
        sell_count = sum(1 for p in sell_patterns if re.search(p, upper))
        if buy_count > sell_count:
            return "BUY"
        elif sell_count > buy_count:
            return "SELL"
        return "HOLD"

    def _update_weights_quartile(self) -> None:
        """Apply quartile-based multiplicative update to weights."""
        # Compute average score per agent over recent trades
        recent = self.trade_history[-50 * len(AGENTS):]  # last ~50 trades per agent
        agent_scores: Dict[str, List[float]] = {a: [] for a in AGENTS}
        for record in recent:
            agent = record["agent"]
            if agent in agent_scores:
                agent_scores[agent].append(record["score"])

        # Need enough data
        counts = [len(v) for v in agent_scores.values()]
        if min(counts) < MIN_TRADES_FOR_QUARTILE:
            return

        avg_scores = {a: sum(s) / len(s) for a, s in agent_scores.items() if s}
        if not avg_scores:
            return

        sorted_agents = sorted(avg_scores.keys(), key=lambda a: avg_scores[a])
        n = len(sorted_agents)
        q1_cutoff = n // 4  # bottom quartile
        q3_cutoff = n - n // 4  # top quartile

        for i, agent in enumerate(sorted_agents):
            if i < q1_cutoff:
                # Bottom quartile → decay
                self.weights[agent] = max(
                    MIN_WEIGHT, self.weights.get(agent, DEFAULT_WEIGHT) * DECAY_FACTOR
                )
            elif i >= q3_cutoff:
                # Top quartile → boost
                self.weights[agent] = min(
                    MAX_WEIGHT, self.weights.get(agent, DEFAULT_WEIGHT) * BOOST_FACTOR
                )
            # Middle → no change

    # ── prompt context formatters ────────────────────────────

    def format_weight_context_for_research_manager(self) -> str:
        bw = self.get_weight("bull_researcher")
        brw = self.get_weight("bear_researcher")
        if bw == DEFAULT_WEIGHT and brw == DEFAULT_WEIGHT:
            return ""
        return f"""
Agent Track Record (weight their arguments proportionally):
- Bull Researcher: reliability weight = {bw:.2f} (range 0.3-2.5, higher = more historically accurate)
- Bear Researcher: reliability weight = {brw:.2f}
Consider these reliability scores when weighing the debate arguments."""

    def format_weight_context_for_risk_manager(self) -> str:
        rw = self.get_weight("risky_analyst")
        sw = self.get_weight("safe_analyst")
        nw = self.get_weight("neutral_analyst")
        if rw == DEFAULT_WEIGHT and sw == DEFAULT_WEIGHT and nw == DEFAULT_WEIGHT:
            return ""
        return f"""
Analyst Track Record (weight their arguments proportionally):
- Risky Analyst: reliability weight = {rw:.2f} (range 0.3-2.5, higher = more historically accurate)
- Safe Analyst: reliability weight = {sw:.2f}
- Neutral Analyst: reliability weight = {nw:.2f}
Consider these reliability scores when weighing the risk debate arguments."""
