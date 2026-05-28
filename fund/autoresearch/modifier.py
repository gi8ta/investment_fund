"""
Prompt modification via LLM for Autoresearch.

Given an underperforming agent's current prompt and trade history,
generates a modified prompt variant for A/B testing.

Adapted from ATLAS-GIC autoresearch prompt optimizer.
"""

from pathlib import Path
from typing import Optional


class PromptModifier:
    """Uses an LLM to generate improved prompt variants."""

    def __init__(self, llm, prompts_dir: str = "fund/prompts"):
        self.llm = llm
        self.prompts_dir = Path(prompts_dir)

    def generate_modification(
        self,
        agent_name: str,
        current_prompt: str,
        performance_summary: str,
        sharpe: float,
    ) -> str:
        """Generate a modified prompt for the underperforming agent.

        Args:
            agent_name: Name of the agent to modify
            current_prompt: Current prompt template text
            performance_summary: Text summary of recent trade performance
            sharpe: Current Sharpe ratio of the agent

        Returns:
            Modified prompt template string (with {placeholders} preserved)
        """
        meta_prompt = f"""You are a prompt engineer optimizing trading agent prompts.

AGENT: {agent_name}
CURRENT SHARPE RATIO: {sharpe:.3f} (this is the worst-performing agent)

PERFORMANCE HISTORY:
{performance_summary}

CURRENT PROMPT:
---
{current_prompt}
---

Your task: Generate an IMPROVED version of this prompt that should lead to better trading decisions.

RULES:
1. Keep ALL {{placeholder}} variables exactly as they are (e.g., {{market_research_report}}, {{history}}, etc.)
2. Do NOT remove any data sources or context variables
3. Focus on improving the REASONING INSTRUCTIONS — how the agent should think about and weigh evidence
4. Consider common failure modes: over-confidence, anchoring bias, ignoring contrary evidence, recency bias
5. The modified prompt should encourage more calibrated, evidence-weighted decisions
6. Keep the prompt roughly the same length (within 20%)
7. Output ONLY the new prompt text, nothing else — no preamble, no explanation

MODIFIED PROMPT:"""

        response = self.llm.invoke(meta_prompt)
        new_prompt = response.content.strip()

        # Basic validation: ensure key placeholders survived
        if not self._validate_placeholders(current_prompt, new_prompt):
            print(f"[DIAG] Autoresearch | modifier | placeholder validation FAILED for {agent_name}, keeping original")
            return current_prompt

        return new_prompt

    def save_candidate(self, agent_name: str, prompt_text: str) -> Path:
        """Save a candidate prompt for A/B testing."""
        path = self.prompts_dir / f"{agent_name}_candidate.txt"
        path.write_text(prompt_text)
        print(f"[DIAG] Autoresearch | modifier | saved candidate: {path}")
        return path

    def promote_candidate(self, agent_name: str) -> bool:
        """Promote a candidate prompt to the active prompt."""
        candidate = self.prompts_dir / f"{agent_name}_candidate.txt"
        active = self.prompts_dir / f"{agent_name}.txt"
        if not candidate.exists():
            return False
        active.write_text(candidate.read_text())
        candidate.unlink()
        print(f"[DIAG] Autoresearch | modifier | promoted {agent_name} candidate to active")
        return True

    def discard_candidate(self, agent_name: str) -> bool:
        """Discard a failed candidate prompt."""
        candidate = self.prompts_dir / f"{agent_name}_candidate.txt"
        if candidate.exists():
            candidate.unlink()
            print(f"[DIAG] Autoresearch | modifier | discarded {agent_name} candidate")
            return True
        return False

    @staticmethod
    def _validate_placeholders(original: str, modified: str) -> bool:
        """Check that all {placeholders} from original are present in modified."""
        import re
        original_placeholders = set(re.findall(r'\{(\w+)\}', original))
        modified_placeholders = set(re.findall(r'\{(\w+)\}', modified))
        # All original placeholders must be in modified
        missing = original_placeholders - modified_placeholders
        if missing:
            print(f"[DIAG] Autoresearch | modifier | missing placeholders: {missing}")
            return False
        return True
