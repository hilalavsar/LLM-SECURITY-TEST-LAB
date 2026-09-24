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


# Fallback list — used only when the API's list_models() call fails.
# Modern (2026) model IDs; the older gemini-1.5-* names are being retired.
KNOWN_GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.8-flash-cyber",
]

# In-process cache so we hit list_models() only once per app lifetime.
_MODEL_CACHE: list[str] | None = None


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

    # Gemini 3.x models are "thinking" models: internal reasoning tokens are
    # billed against max_output_tokens, and they routinely burn 300-400 of them
    # before emitting a single visible character. With Ollama's num_predict
    # (320) the whole budget went to reasoning and responses came back cut in
    # half — silently, with finish_reason=MAX_TOKENS. Anything below this floor
    # produces fragments, so we raise it for Gemini only.
    MIN_OUTPUT_TOKENS = 2048

    def _config(self, options: dict) -> dict:
        # Map our generic options to Gemini's GenerationConfig fields.
        return {
            "temperature": options.get("temperature", 0),
            "max_output_tokens": max(
                options.get("num_predict", 320), self.MIN_OUTPUT_TOKENS
            ),
        }

    @staticmethod
    def _truncation_error(resp) -> str | None:
        """Return an error string if the model was cut off mid-answer.

        A truncated response must never be scored: a fragment can hide a leak
        the model was about to produce (false PASS) or clip a refusal down to
        something that looks like one (false FAIL). Both happened on the
        gemini-3.5-flash run of 2026-09-03 before this check existed.
        """
        try:
            cand = resp.candidates[0]
        except (AttributeError, IndexError):
            return None
        # FinishReason: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION.
        reason = int(getattr(cand, "finish_reason", 1) or 1)
        if reason == 2:
            return ("response truncated (finish_reason=MAX_TOKENS) — thinking "
                    "tokens consumed the output budget; raise max_output_tokens")
        if reason == 3:
            return "response blocked by provider safety filter (finish_reason=SAFETY)"
        return None

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
                text, self.model_name, (time.time() - t0) * 1000, options,
                self._truncation_error(resp),
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
                text, self.model_name, (time.time() - t0) * 1000, options,
                self._truncation_error(resp),
            )
        except Exception as e:  # noqa: BLE001
            return GenerationResult(
                "", self.model_name, (time.time() - t0) * 1000, options, str(e)
            )

    def health_check(self) -> bool:
        return _GEMINI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY"))

    @staticmethod
    def list_models() -> list[str]:
        """Discover Gemini models actually available to this API key.

        Calls the API once and caches the result. Falls back to the hardcoded
        modern-name list if the discovery call fails (e.g. offline, quota).
        Only models that support `generateContent` are returned.
        """
        global _MODEL_CACHE
        if not _GEMINI_AVAILABLE or not os.getenv("GEMINI_API_KEY"):
            return []
        if _MODEL_CACHE is not None:
            return list(_MODEL_CACHE)
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            discovered: list[str] = []
            for m in genai.list_models():
                if "generateContent" not in getattr(m, "supported_generation_methods", []):
                    continue
                # Strip the 'models/' prefix the API prepends.
                name = m.name.split("/", 1)[-1] if m.name.startswith("models/") else m.name
                # Skip embedding / aqa / preview / experimental variants — the
                # -exp and -preview tags often return 404 on generateContent
                # even though list_models advertises them.
                if any(skip in name for skip in (
                    "embedding", "aqa", "learnlm",
                    "-exp", "-experimental", "-preview", "-tuning", "-vision-latest",
                    "-image",  # image-generation variants — not text targets
                    "-live",   # WebSocket-only Live API, rejects generateContent
                    "robotics", "computer-use", "deep-research",
                )):
                    continue
                discovered.append(name)
            # Sort so 2.5-pro comes first, then 2.5-flash, etc.
            discovered.sort(key=lambda n: (n.count("-"), n))
            _MODEL_CACHE = discovered or list(KNOWN_GEMINI_MODELS)
            return list(_MODEL_CACHE)
        except Exception:  # noqa: BLE001
            _MODEL_CACHE = list(KNOWN_GEMINI_MODELS)
            return list(_MODEL_CACHE)
