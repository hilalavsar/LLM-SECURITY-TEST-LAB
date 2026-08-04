"""L2 · MockAdapter — deterministik sahte model.

Neden var: LLM cevaplari yavas ve degisken. Evaluator'in dogru
calisip calismadigini test etmek icin BILINEN, sabit cevaplar
donduren bir sahte model gerekir. Boylece "regex'im FAIL veriyor mu"
sorusunu saniyede test ederiz — gercek modele gerek kalmadan.

Bu dosya HENUZ BOS bir iskelet.
"""

from __future__ import annotations
from .base import BaseModelAdapter, GenerationResult


class MockAdapter(BaseModelAdapter):
    model_name = "mock"

    def generate(self, prompt: str, options: dict) -> GenerationResult:
        raise NotImplementedError("Birlikte yazacagiz")

    def health_check(self) -> bool:
        return True
