"""L2 · Model Abstraction Layer — Adapter sozlesmesi (interface).

Test runner sadece `generate(prompt, options)` cagirir; hangi modele
gittigini bilmez. Priz adaptoru mantigi: her model runtime'i (Ollama,
uzak API, mock) bu sozlesmeyi uygular.

Bu dosya HENUZ BOS bir iskelet. Implementasyonu birlikte yazacagiz.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Bir model cagrisinin ham sonucu + olcumler.

    Reproduksiyon icin uretim ayarlari da (options) burada saklanir.
    """
    text: str
    model_name: str
    latency_ms: float
    options: dict
    error: str | None = None


class BaseModelAdapter(ABC):
    """Tum adapter'larin uyacagi sozlesme."""

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, options: dict) -> GenerationResult:
        """Prompt gonder, modelin cevabini + olcumleri dondur."""
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """Runtime ayakta mi? (or. Ollama daemon calisiyor mu)"""
        raise NotImplementedError
