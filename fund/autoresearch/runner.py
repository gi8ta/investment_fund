"""
Autoresearch A/B test runner for Investment Fund.

Orchestrates the autoresearch loop:
1. Identify worst-performing agent via scorecard
2. Generate modified prompt via modifier
3. (Future) Run A/B test comparing original vs modified over N signals
4. Promote or discard based on Sharpe improvement

This module provides the orchestration logic; actual A/B testing requires
running the full pipeline multiple times and is triggered from backtest.py.

Adapted from ATLAS-GIC autoresearch loop.
"""

from pathlib import Path
from typing import Optional, Dict

from .scorecard import AgentScorecard
from .modifier import PromptModifier


class AutoresearchRunner:
    """Orchestrates the autoresearch prompt optimization loop."""

    def __init__(
        self,
        llm,
        scorecard: AgentScorecard,
        prompts_dir: str = "fund/prompts",
        sharpe_threshold: float = 0.0,
    ):
        self.scorecard = scorecard
        self.modifier = PromptModifier(llm, prompts_dir)
        self.prompts_dir = Path(prompts_dir)
        self.sharpe_threshold = sharpe_threshold
        self._pending_ab: Optional[Dict] = None  # tracks in-progress A/B test

    def check_and_trigger(self, signal_count: int, trigger_every: int = 20) -> Optional[str]:
        """Check if autoresearch should be triggered.

        Called after each signal. If signal_count is a multiple of
        trigger_every, identifies the weakest agent and generates a
        candidate prompt modification.

        Args:
            signal_count: Total signals processed so far
            trigger_every: Trigger interval

        Returns:
            Agent name being modified, or None if not triggered.
        """
        if signal_count % trigger_every != 0 or signal_count == 0:
            return None

        # Skip if an A/B test is already in progress
        if self._pending_ab is not None:
            print("[DIAG] Autoresearch | runner | A/B test already in progress, skipping")
            return None

        result = self.scorecard.find_weakest_agent()
        if result is None:
            print("[DIAG] Autoresearch | runner | insufficient data, skipping")
            return None

        agent_name, sharpe = result
        print(f"[DIAG] Autoresearch | runner | weakest agent: {agent_name} (Sharpe={sharpe:.3f})")

        # Only modify if Sharpe is below threshold
        if sharpe > self.sharpe_threshold:
            print(f"[DIAG] Autoresearch | runner | Sharpe {sharpe:.3f} > threshold {self.sharpe_threshold}, skipping")
            return None

        # Load current prompt
        prompt_file = self.prompts_dir / f"{agent_name}.txt"
        if not prompt_file.exists():
            print(f"[DIAG] Autoresearch | runner | no prompt file for {agent_name}, skipping")
            return None

        current_prompt = prompt_file.read_text()
        perf_summary = self.scorecard.get_history_summary(agent_name)

        # Generate modified prompt
        new_prompt = self.modifier.generate_modification(
            agent_name, current_prompt, perf_summary, sharpe
        )

        if new_prompt == current_prompt:
            print(f"[DIAG] Autoresearch | runner | modification failed for {agent_name}")
            return None

        # Save candidate and start tracking A/B test
        self.modifier.save_candidate(agent_name, new_prompt)
        self._pending_ab = {
            "agent": agent_name,
            "original_sharpe": sharpe,
            "signals_at_start": signal_count,
            "ab_signals": 0,
            "ab_target": 5,  # compare over 5 signals
        }

        print(f"[DIAG] Autoresearch | runner | started A/B test for {agent_name}")
        return agent_name

    def record_ab_signal(self, actual_return_pct: float, signal: str) -> Optional[str]:
        """Record a signal during an active A/B test.

        Returns:
            "promoted" if candidate beat original and was promoted,
            "discarded" if candidate lost and was discarded,
            None if A/B test still in progress.
        """
        if self._pending_ab is None:
            return None

        self._pending_ab["ab_signals"] += 1

        if self._pending_ab["ab_signals"] < self._pending_ab["ab_target"]:
            return None

        # A/B test complete — evaluate
        agent_name = self._pending_ab["agent"]
        original_sharpe = self._pending_ab["original_sharpe"]
        new_sharpe = self.scorecard.compute_sharpe(agent_name)

        print(f"[DIAG] Autoresearch | runner | A/B complete for {agent_name}: "
              f"original_sharpe={original_sharpe:.3f}, new_sharpe={new_sharpe}")

        if new_sharpe is not None and new_sharpe > original_sharpe:
            self.modifier.promote_candidate(agent_name)
            self._pending_ab = None
            return "promoted"
        else:
            self.modifier.discard_candidate(agent_name)
            self._pending_ab = None
            return "discarded"

    @property
    def is_ab_active(self) -> bool:
        return self._pending_ab is not None

    @property
    def ab_agent(self) -> Optional[str]:
        return self._pending_ab["agent"] if self._pending_ab else None
