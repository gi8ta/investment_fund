"""
Prompt loading utilities for Investment Fund.

Agents can load their prompts from external .txt files (for autoresearch
modification) with a fallback to their inline default strings.
"""

from pathlib import Path
from typing import Optional

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(agent_name: str, prompts_dir: Optional[str] = None) -> Optional[str]:
    """Load a prompt template from file.

    Looks for ``<agent_name>.txt`` in *prompts_dir* (or the default prompts
    directory shipped with the package).  Returns ``None`` if the file does not
    exist, letting the caller fall back to its inline default.

    The returned string contains ``{placeholder}`` markers suitable for
    ``str.format(**kwargs)`` interpolation.
    """
    base = Path(prompts_dir) if prompts_dir else _PROMPTS_DIR
    path = base / f"{agent_name}.txt"
    if path.exists():
        return path.read_text()
    return None


def load_prompt_or_default(agent_name: str, default: str, prompts_dir: Optional[str] = None) -> str:
    """Load prompt from file, falling back to *default* inline string."""
    loaded = load_prompt(agent_name, prompts_dir)
    return loaded if loaded is not None else default
