"""L2 · GeminiAdapter — Google AI Studio (Gemini) adapter.

Implements the same BaseModelAdapter contract as OllamaAdapter, so the runner
and views can treat Gemini models exactly like local Ollama models. The only
difference is the network path: Gemini goes over HTTPS to Google's endpoint
instead of localhost.

Configuration:
    GEMINI_API_KEY  — Google AI Studio key (https://aistudio.google.com/)
    Model names   — bare strings ("gemini-1.5-pro"), no prefix here. The
                    runner tags them with a "gemini:" prefix in the UI to keep
                    Ollama and Gemini names visually separated.
"""

from __future__ import annotations

import os
import time

from .base import BaseModelAdapter, GenerationResult

try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# Curated list — extend as new models ship. Kept short on purpose so the
# target dropdown stays clean.
KNOWN_GEMINI_MODELS = [
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-flash-exp",
]


class GeminiAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, api_key: str | None = None):
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai kütüphanesi yüklü değil. "
                "`pip install google-generativeai` çalıştır."
            )
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY ortam değişkeni tanımlı değil.")
        self.model_name = model_name
        genai.configure(api_key=key)

    def _config(self, options: dict) -> dict:
        # Map our generic options to Gemini's GenerationConfig fields.
        return {
            "temperature": options.get("temperature", 0),
            "max_output_tokens": options.get("num_predict", 320),
        }

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        options: dict | None = None,
    ) -> GenerationResult:
        options = options or {}
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt or None,
            generation_config=self._config(options),
        )
        t0 = time.time()
        try:
            resp = model.generate_content(user_prompt)
            # Gemini returns no .text when safety-blocks the response — treat
            # as empty and let evaluator handle it.
            text = getattr(resp, "text", "") or ""
            return GenerationResult(
                text, self.model_name, (time.time() - t0) * 1000, options
            )
        except Exception as e:  # noqa: BLE001
            return GenerationResult(
                "", self.model_name, (time.time() - t0) * 1000, options, str(e)
            )

    def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        options: dict | None = None,
    ) -> GenerationResult:
        options = options or {}
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system_prompt or None,
            generation_config=self._config(options),
        )
        # Gemini uses 'model' for assistant and expects {'role','parts':[str]}.
        history = [
            {"role": "model" if m["role"] == "assistant" else m["role"],
             "parts": [m["content"]]}
            for m in messages[:-1]
        ]
        last_user = messages[-1]["content"]
        t0 = time.time()
        try:
            chat_session = model.start_chat(history=history)
            resp = chat_session.send_message(last_user)
            text = getattr(resp, "text", "") or ""
            return GenerationResult(
                text, self.model_name, (time.time() - t0) * 1000, options
            )
        except Exception as e:  # noqa: BLE001
            return GenerationResult(
                "", self.model_name, (time.time() - t0) * 1000, options, str(e)
            )

    def health_check(self) -> bool:
        return _GEMINI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def list_models() -> list[str]:
        """Return the curated Gemini model list; empty if the SDK/key missing."""
        if not _GEMINI_AVAILABLE or not os.getenv("GEMINI_API_KEY"):
            return []
        return list(KNOWN_GEMINI_MODELS)
