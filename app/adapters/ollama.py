"""L2 · OllamaAdapter — HTTP calls to the local Ollama daemon.

Uses POST {host}/api/chat so the system prompt (defense) and user prompt
(attack) go in as separate messages — which is exactly the boundary we test.
Uses the stdlib (urllib) to avoid extra dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.request

from .base import BaseModelAdapter, GenerationResult


class OllamaAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host.rstrip("/")

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        options: dict | None = None,
    ) -> GenerationResult:
        options = options or {}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        body = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=200) as r:
                data = json.loads(r.read())
            text = data.get("message", {}).get("content", "")
            return GenerationResult(text, self.model_name, (time.time() - t0) * 1000, options)
        except Exception as e:  # noqa: BLE001 - capture any runtime error safely
            return GenerationResult("", self.model_name, (time.time() - t0) * 1000, options, str(e))

    def health_check(self) -> bool:
        try:
            urllib.request.urlopen(self.host + "/api/version", timeout=3)
            return True
        except Exception:
            return False

    @staticmethod
    def list_models(host: str = "http://localhost:11434") -> list[str]:
        """Return the names of models installed in the Ollama daemon."""
        try:
            with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=5) as r:
                data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
