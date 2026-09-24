"""OpenAI-compatible adapter + runtime provider registry.

Runs fully offline: a tiny stub server on 127.0.0.1 plays the provider, so no
API key, internet, Ollama, or PostgreSQL is needed.

    pytest tests/test_openai_compat.py -v
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app import providers
from app.adapters.openai_compat import OpenAICompatAdapter

KEY = "good-key-1234567890abcdef"
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 320}


class _Stub(BaseHTTPRequestHandler):
    bodies: list[dict] = []

    def log_message(self, *args):  # keep pytest output clean
        pass

    def _send(self, code: int, obj) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authed(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {KEY}"

    def do_GET(self):
        if not self._authed():
            return self._send(401, {"error": {"message": "invalid api key"}})
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "m-ok"}, {"id": "vendor/m:free"}]})
        if self.path == "/gemini/models":
            # Gemini-style listing: mixed id prefixes plus non-chat models.
            return self._send(200, {"data": [
                {"id": "models/gemini-2.5-flash"},
                {"id": "gemini-2.5-flash-lite"},
                {"id": "models/text-embedding-004"},
                {"id": "imagen-4.0-generate-001"},
            ]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.bodies.append(body)
        if not self._authed():
            return self._send(401, {"error": {"message": "invalid api key"}})
        model = body["model"]
        if model == "m-no-seed" and "seed" in body:
            # Mistral-style validation error: no "message", just a detail list.
            return self._send(422, {"detail": [
                {"loc": ["body", "seed"], "msg": "Extra inputs are not permitted"}]})
        if model == "m-reasoning" and "max_tokens" in body:
            return self._send(400, {"error": {"message":
                "Unsupported parameter: 'max_tokens' is not supported with this "
                "model. Use 'max_completion_tokens' instead."}})
        if model == "m-leaky-error":
            return self._send(500, {"error": {"message":
                f"upstream failed, header was {self.headers.get('Authorization')}"}})
        if model == "m-trunc":
            return self._send(200, {"choices": [
                {"message": {"content": "partial ans"}, "finish_reason": "length"}]})
        if model == "m-refusal":
            return self._send(200, {"choices": [
                {"message": {"content": None, "refusal": "I can't help with that."},
                 "finish_reason": "stop"}]})
        self._send(200, {"choices": [
            {"message": {"content": "hello"}, "finish_reason": "stop"}]})


@pytest.fixture()
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Stub.bodies.clear()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()
    providers.clear_providers()


# --- adapter ----------------------------------------------------------------

def test_generate_sends_system_and_user_and_returns_text(base_url):
    res = OpenAICompatAdapter("m-ok", base_url, KEY).generate("hi", "SYS", OPTIONS)
    assert res.error is None and res.text == "hello"
    body = _Stub.bodies[-1]
    assert body["messages"] == [{"role": "system", "content": "SYS"},
                                {"role": "user", "content": "hi"}]
    assert body["seed"] == 42
    assert body["max_tokens"] >= OpenAICompatAdapter.MIN_OUTPUT_TOKENS


def test_truncated_answer_is_reported_as_error(base_url):
    res = OpenAICompatAdapter("m-trunc", base_url, KEY).generate("hi", "SYS", OPTIONS)
    assert res.error and "truncated" in res.error


def test_refusal_field_counts_as_the_reply(base_url):
    res = OpenAICompatAdapter("m-refusal", base_url, KEY).generate("hi", None, OPTIONS)
    assert res.error is None
    assert res.text == "I can't help with that."


def test_rejected_seed_is_dropped_and_retried(base_url):
    res = OpenAICompatAdapter("m-no-seed", base_url, KEY).generate("hi", None, OPTIONS)
    assert res.error is None and res.text == "hello"
    assert "seed" in _Stub.bodies[0] and "seed" not in _Stub.bodies[-1]


def test_max_tokens_renamed_for_reasoning_models(base_url):
    res = OpenAICompatAdapter("m-reasoning", base_url, KEY).generate("hi", None, OPTIONS)
    assert res.error is None
    assert "max_completion_tokens" in _Stub.bodies[-1]
    assert "max_tokens" not in _Stub.bodies[-1]


def test_error_text_never_contains_the_key(base_url):
    res = OpenAICompatAdapter("m-leaky-error", base_url, KEY).generate("hi", None, OPTIONS)
    assert res.error and KEY not in res.error


def test_chat_keeps_history_order(base_url):
    history = [{"role": "user", "content": "a"},
               {"role": "assistant", "content": "b"},
               {"role": "user", "content": "c"}]
    OpenAICompatAdapter("m-ok", base_url, KEY).chat(history, "SYS", OPTIONS)
    assert [m["content"] for m in _Stub.bodies[-1]["messages"]] == ["SYS", "a", "b", "c"]


# --- registry ---------------------------------------------------------------

def test_add_provider_discovers_models_and_hides_key(base_url):
    p = providers.add_provider("Test Sağlayıcı", base_url, KEY)
    assert p.id == "test-saglayici"
    assert p.models == ["m-ok", "vendor/m:free"]
    assert KEY not in p.masked_key and KEY not in repr(p)
    values = [c["value"] for c in providers.model_choices()]
    assert "api:test-saglayici:vendor/m:free" in values


def test_wrong_key_is_rejected_before_registering(base_url):
    with pytest.raises(providers.ProviderError, match="401"):
        providers.add_provider("Bad", base_url, "wrong-key-0000000000")
    assert providers.list_providers() == []


def test_unknown_manual_model_is_rejected(base_url):
    with pytest.raises(providers.ProviderError, match="tanımıyor"):
        providers.add_provider("X", base_url, KEY, "m-ok, typo-model")


def test_model_value_keeps_colons_and_slashes():
    assert providers.parse_model_value("api:groq:vendor/model:free") == ("groq", "vendor/model:free")
    assert providers.parse_model_value("gemini:gemini-2.5-flash") is None
    assert providers.parse_model_value("api:no-model") is None


def test_adapter_for_forgotten_provider_explains_restart():
    with pytest.raises(providers.ProviderError, match="yeniden başlayınca"):
        providers.build_adapter("api:gone:some-model")


@pytest.mark.parametrize("url", [
    "ftp://example.com/v1",
    "https://user:pass@example.com/v1",
    "https://example.com/v1?key=abc",
    "not a url",
])
def test_bad_base_urls_are_rejected(url):
    with pytest.raises(providers.ProviderError):
        providers.validate_base_url(url)


def test_key_never_sent_over_plain_http_to_remote_host():
    with pytest.raises(providers.ProviderError, match="https"):
        providers.add_provider("Remote", "http://example.com/v1", KEY)


def test_duplicate_names_get_unique_ids(base_url):
    a = providers.add_provider("Groq", base_url, KEY)
    b = providers.add_provider("Groq", base_url, KEY)
    assert (a.id, b.id) == ("groq", "groq-2")


def test_redact_known_covers_registered_keys(base_url):
    providers.add_provider("Groq", base_url, KEY)
    assert KEY not in providers.redact_known(f"boom {KEY} boom")


# --- Gemini through its OpenAI-compatible endpoint --------------------------

def test_gemini_listing_is_normalized_and_non_chat_models_skipped(base_url):
    p = providers.add_provider("Gemini", base_url.replace("/v1", "/gemini"), KEY)
    assert p.models == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]


def test_bare_gemini_name_matches_prefixed_listing(base_url):
    p = providers.add_provider("Gemini", base_url.replace("/v1", "/gemini"), KEY,
                               "gemini-2.5-flash")
    assert p.models == ["gemini-2.5-flash"]


def test_gemini_preset_points_at_openai_compat_endpoint():
    gemini = next(p for p in providers.PRESETS if p["name"] == "Gemini")
    assert providers.validate_base_url(gemini["base_url"]).endswith("/v1beta/openai")
