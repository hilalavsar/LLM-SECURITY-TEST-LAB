"""L2 · OpenAICompatAdapter — any server that speaks the OpenAI chat API.

Most of the ecosystem exposes the same `POST {base_url}/chat/completions`
shape, so one adapter covers OpenAI, Groq, OpenRouter, DeepSeek, Mistral,
Together, and local servers such as LM Studio or vLLM. Endpoints and keys are
registered at runtime from the UI (app/providers.py); this module never reads
keys from disk.

Stdlib only (urllib), same as OllamaAdapter.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from app.redact import redact

from .base import BaseModelAdapter, GenerationResult

# Some providers sit behind bot filters that reject urllib's default agent.
_USER_AGENT = "llm-security-test-lab/0.4"
_TIMEOUT_S = 200
# Fallback budget when a server says the requested output is too long for it.
_SMALL_BUDGET = 512


class OpenAICompatAdapter(BaseModelAdapter):
    # Same lesson as GeminiAdapter: reasoning models pay for hidden thinking out
    # of the output budget, and Ollama's num_predict (320) cuts their answers
    # off. Billing counts generated tokens, not the ceiling, so a high floor
    # costs nothing when the model stops early.
    MIN_OUTPUT_TOKENS = 2048

    def __init__(self, model_name: str, base_url: str, api_key: str = ""):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    def generate(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        options: dict | None = None,
    ) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._complete(messages, options or {})

    def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        options: dict | None = None,
    ) -> GenerationResult:
        full = [{"role": "system", "content": system_prompt}] if system_prompt else []
        full.extend(messages)
        return self._complete(full, options or {})

    def health_check(self) -> bool:
        try:
            self.list_models(self.base_url, self._api_key)
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def list_models(base_url: str, api_key: str = "") -> list[str]:
        """GET {base_url}/models -> sorted model ids. Raises on any failure."""
        req = urllib.request.Request(
            base_url.rstrip("/") + "/models", headers=_headers(api_key)
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        items = data.get("data", []) if isinstance(data, dict) else data
        ids = [m["id"] for m in items if isinstance(m, dict) and m.get("id")]
        # Google's native API names models "models/gemini-2.5-flash" while chat
        # requests take the bare name. Strip the prefix so listings and
        # hand-typed names always compare equal.
        return sorted({i.removeprefix("models/") for i in ids})

    # --- internals ------------------------------------------------------

    def _complete(self, messages: list[dict], options: dict) -> GenerationResult:
        body = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "temperature": options.get("temperature", 0),
            "max_tokens": max(options.get("num_predict", 320), self.MIN_OUTPUT_TOKENS),
        }
        if "seed" in options:
            body["seed"] = options["seed"]

        t0 = time.time()
        try:
            data = self._post_with_fallbacks(body)
            choices = data.get("choices") if isinstance(data, dict) else None
            if not choices:
                raise RuntimeError(f"unexpected response: {json.dumps(data)[:200]}")
            choice = choices[0]
        except Exception as e:  # noqa: BLE001
            return GenerationResult(
                "", self.model_name, _ms(t0), options, redact(str(e), [self._api_key])
            )
        return GenerationResult(
            _text_of(choice.get("message") or {}), self.model_name, _ms(t0),
            options, _finish_error(choice.get("finish_reason")),
        )

    def _post_with_fallbacks(self, body: dict) -> dict:
        """POST chat/completions, adjusting parameters providers commonly reject.

        The OpenAI shape is a de facto standard, not a spec: providers return
        400/422 for different fields. Each retry changes exactly one field the
        error names:
          max_tokens rejected, max_completion_tokens suggested -> rename
          output budget too large for the model               -> shrink to 512
          seed not accepted                                   -> drop
          temperature not accepted                            -> drop
        Dropping seed or temperature weakens reproducibility for that provider,
        which beats failing every case in the run.
        """
        body = dict(body)
        for _ in range(5):
            try:
                return self._post(body)
            except urllib.error.HTTPError as e:
                detail = _read_error(e)
                if e.code not in (400, 422) or not _adjust(body, detail.lower()):
                    raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        raise RuntimeError("request still rejected after parameter fallbacks")

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=_headers(self._api_key),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:
            return json.loads(r.read())


def _adjust(body: dict, low: str) -> bool:
    """Apply one fix for a 400/422 error message. False = nothing left to try."""
    if "max_tokens" in body and "max_completion_tokens" in low:
        body["max_completion_tokens"] = body.pop("max_tokens")
        return True
    budget_key = "max_tokens" if "max_tokens" in body else "max_completion_tokens"
    if (budget_key in body and body[budget_key] > _SMALL_BUDGET
            and ("context length" in low or "maximum context" in low or budget_key in low)):
        body[budget_key] = _SMALL_BUDGET
        return True
    if "seed" in body and "seed" in low:
        body.pop("seed")
        return True
    if "temperature" in body and "temperature" in low:
        body.pop("temperature")
        return True
    return False


def _headers(api_key: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _read_error(e: urllib.error.HTTPError) -> str:
    try:
        raw = e.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return str(e.reason or "")
    try:
        obj = json.loads(raw)
    except ValueError:
        return raw[:300]
    err = obj.get("error", obj) if isinstance(obj, dict) else obj
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])[:300]
    return json.dumps(err)[:300]


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        # Some providers return typed parts (e.g. separate thinking blocks);
        # only the visible text parts are the model's answer.
        content = "".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type", "text") == "text"
        )
    # OpenAI puts refusals in a separate field with content=None. A refusal is
    # the model's answer and has to be scored like any other reply.
    return content or message.get("refusal") or ""


def _finish_error(reason: str | None) -> str | None:
    # Same rule as the Gemini adapter: an incomplete answer is never scored.
    if reason == "length":
        return ("response truncated (finish_reason=length) — output budget "
                "exhausted before the answer ended")
    if reason == "content_filter":
        return "response blocked by provider content filter (finish_reason=content_filter)"
    return None


def _ms(t0: float) -> float:
    return (time.time() - t0) * 1000
