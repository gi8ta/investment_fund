"""
JANUS Meta-Weighting for Investment Fund.

Runs multiple TradingGraph configurations ("cohorts") in parallel,
tracks their individual performance (hit rate + Sharpe), and blends their
signals via softmax weighting to produce a final combined signal.

Adapted from ATLAS-GIC janus.py.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ── Constants ────────────────────────────────────────────────
MIN_WEIGHT = 0.2
MAX_WEIGHT = 0.8
ROLLING_WINDOW = 30
SOFTMAX_TEMPERATURE = 1.0


@dataclass
class CohortConfig:
    """Configuration override for a single JANUS cohort."""
    name: str
    overrides: Dict = field(default_factory=dict)


@dataclass
class CohortRecord:
    """Performance record for a single trade within a cohort."""
    ticker: str
    trade_date: str
    signal: str  # BUY / SELL / HOLD
    actual_return_pct: float = 0.0
    hit: bool = False  # True if signal direction matched return direction


class JanusMetaWeighter:
    """Manages multiple cohorts and blends their signals."""

    def __init__(
        self,
        cohort_configs: List[CohortConfig],
        state_file: str = "janus_state.json",
    ):
        self.cohort_configs = cohort_configs
        self.state_file = Path(state_file)

        # Performance tracking per cohort
        self.history: Dict[str, List[CohortRecord]] = {
            c.name: [] for c in cohort_configs
        }
        self.weights: Dict[str, float] = {
            c.name: 1.0 / len(cohort_configs) for c in cohort_configs
        }

        self._load_state()

    # ── Persistence ──────────────────────────────────────────

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            data = json.loads(self.state_file.read_text())
            self.weights = data.get("weights", self.weights)
            for name, records in data.get("history", {}).items():
                if name in self.history:
                    self.history[name] = [
                        CohortRecord(**r) for r in records[-ROLLING_WINDOW * 3:]
                    ]
        except (json.JSONDecodeError, IOError, TypeError):
            pass

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "history": {
                name: [asdict(r) for r in records[-ROLLING_WINDOW * 3:]]
                for name, records in self.history.items()
            },
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    # ── Signal collection ────────────────────────────────────

    def record_signal(
        self,
        cohort_name: str,
        ticker: str,
        trade_date: str,
        signal: str,
    ) -> None:
        """Record a signal from a cohort (before outcome is known)."""
        if cohort_name not in self.history:
            return
        self.history[cohort_name].append(CohortRecord(
            ticker=ticker,
            trade_date=trade_date,
            signal=signal,
        ))

    def record_outcome(
        self,
        cohort_name: str,
        ticker: str,
        trade_date: str,
        actual_return_pct: float,
    ) -> None:
        """Update a previously recorded signal with the actual outcome."""
        if cohort_name not in self.history:
            return
        for rec in reversed(self.history[cohort_name]):
            if rec.ticker == ticker and rec.trade_date == trade_date:
                rec.actual_return_pct = actual_return_pct
                # A hit: BUY + positive return, or SELL + negative return
                if rec.signal == "BUY":
                    rec.hit = actual_return_pct > 0
                elif rec.signal == "SELL":
                    rec.hit = actual_return_pct < 0
                else:
                    rec.hit = False  # HOLD is never a "hit"
                break

    # ── Weight computation ───────────────────────────────────

    def update_weights(self) -> Dict[str, float]:
        """Recompute cohort weights based on recent performance.

        Uses combined score = 0.5 * hit_rate + 0.5 * normalized_sharpe,
        then applies softmax with bounds [MIN_WEIGHT, MAX_WEIGHT].
        """
        scores: Dict[str, float] = {}

        for name in self.history:
            recent = [r for r in self.history[name][-ROLLING_WINDOW:]
                      if r.actual_return_pct != 0.0 or r.signal != "HOLD"]
            if len(recent) < 3:
                scores[name] = 0.5  # neutral score when insufficient data
                continue

            # Hit rate
            active = [r for r in recent if r.signal != "HOLD"]
            hit_rate = sum(1 for r in active if r.hit) / max(len(active), 1)

            # Sharpe ratio
            returns = []
            for r in recent:
                if r.signal == "BUY":
                    returns.append(r.actual_return_pct)
                elif r.signal == "SELL":
                    returns.append(-r.actual_return_pct)
                # HOLD contributes 0
            if returns:
                mean_r = sum(returns) / len(returns)
                var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                std = math.sqrt(var) if var > 0 else 1e-8
                sharpe = mean_r / std
            else:
                sharpe = 0.0

            # Normalize sharpe to [0, 1] range (clip extreme values)
            norm_sharpe = max(0.0, min(1.0, (sharpe + 2.0) / 4.0))

            scores[name] = 0.5 * hit_rate + 0.5 * norm_sharpe

        # Softmax with temperature
        if not scores:
            return self.weights

        max_score = max(scores.values())
        exp_scores = {
            name: math.exp((s - max_score) / SOFTMAX_TEMPERATURE)
            for name, s in scores.items()
        }
        total = sum(exp_scores.values())
        raw_weights = {name: e / total for name, e in exp_scores.items()}

        # Apply bounds
        self.weights = {
            name: max(MIN_WEIGHT, min(MAX_WEIGHT, w))
            for name, w in raw_weights.items()
        }

        # Re-normalize so weights sum to 1
        wsum = sum(self.weights.values())
        if wsum > 0:
            self.weights = {n: w / wsum for n, w in self.weights.items()}

        self._save_state()

        print(f"[DIAG] JANUS | updated weights: {self.weights} | scores: {scores}")
        return dict(self.weights)

    # ── Signal blending ──────────────────────────────────────

    def blend_signals(
        self,
        cohort_signals: Dict[str, str],
    ) -> Tuple[str, Dict[str, float]]:
        """Blend signals from multiple cohorts using current weights.

        Args:
            cohort_signals: {cohort_name: "BUY"/"SELL"/"HOLD"}

        Returns:
            (blended_signal, vote_breakdown) where vote_breakdown shows
            the weighted votes for each direction.
        """
        direction_votes: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}

        for name, signal in cohort_signals.items():
            w = self.weights.get(name, 1.0 / len(self.cohort_configs))
            if signal in direction_votes:
                direction_votes[signal] += w

        # Winner is the direction with most weighted votes
        winner = max(direction_votes, key=lambda d: direction_votes[d])

        # If HOLD ties with another direction, prefer the other direction
        if winner == "HOLD" and direction_votes["HOLD"] < 0.5:
            non_hold = {d: v for d, v in direction_votes.items() if d != "HOLD"}
            if non_hold:
                alt = max(non_hold, key=lambda d: non_hold[d])
                if non_hold[alt] > direction_votes["HOLD"]:
                    winner = alt

        print(f"[DIAG] JANUS | blend: {cohort_signals} → {winner} | votes={direction_votes}")
        return winner, direction_votes

    def get_cohort_config(self, base_config: dict, cohort_name: str) -> dict:
        """Create a full config dict for a specific cohort.

        Starts from base_config and applies the cohort's overrides.
        """
        import copy
        config = copy.deepcopy(base_config)
        for cc in self.cohort_configs:
            if cc.name == cohort_name:
                config.update(cc.overrides)
                break
        return config
