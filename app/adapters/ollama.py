"""L2 · OllamaAdapter — yerel Ollama daemon'a HTTP cagrisi.

Endpoint: POST {OLLAMA_HOST}/api/generate
Ayarlar (temperature, seed, top_p, num_predict) request JSON'unun
"options" alaninda gider — masaustu uygulamasindan DEGIL, buradan.

Bu dosya HENUZ BOS bir iskelet. smoke_test.py calisan bir ornek icerir;
bu adapter onu duzgun bir sinifa donusturecek.
"""

from __future__ import annotations
from .base import BaseModelAdapter, GenerationResult


class OllamaAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate(self, prompt: str, options: dict) -> GenerationResult:
        raise NotImplementedError("Birlikte yazacagiz")

    def health_check(self) -> bool:
        raise NotImplementedError("Birlikte yazacagiz")
