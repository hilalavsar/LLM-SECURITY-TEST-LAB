"""L2 · Model abstraction layer — the adapter contract (interface).

The runner only calls `generate(...)`; it does not know which runtime it
talks to. Power-adapter analogy: every model runtime (Ollama, remote API,
mock) implements this one contract.

For system-prompt testing we pass BOTH a system prompt (the defense) and a
user prompt (the attack), so `generate` accepts an optional system_prompt.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Raw result of a single model call plus measurements.

    Generation options are kept here too, for reproducibility.
    """

    text: str
    model_name: str
    latency_ms: float
    options: dict
    error: str | None = None


class BaseModelAdapter(ABC):
    """Contract every adapter must follow."""

    model_name: str

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        options: dict | None = None,
    ) -> GenerationResult:
        """Send system+user prompts, return the model's reply and metrics."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Is the runtime up? (e.g. Ollama daemon running)"""
        raise NotImplementedError
