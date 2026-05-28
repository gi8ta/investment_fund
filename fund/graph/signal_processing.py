# fund/graph/signal_processing.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (BUY, SELL, or HOLD)
        """
        print(f"[DIAG] SignalProcessor | START | raw_signal_len={len(full_signal)} | snippet={full_signal[:500]!r}")

        # CRO override takes precedence — no need to ask LLM
        if "CRO OVERRIDE TO HOLD" in full_signal.upper():
            print("[DIAG] SignalProcessor | DONE | extracted='HOLD' (CRO override detected)")
            return "HOLD"

        messages = [
            (
                "system",
                "You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts. Your task is to extract the investment decision: SELL, BUY, or HOLD. Provide only the extracted decision (SELL, BUY, or HOLD) as your output, without adding any additional text or information.",
            ),
            ("human", full_signal),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        print(f"[DIAG] SignalProcessor | DONE | extracted={result!r}")
        return result
