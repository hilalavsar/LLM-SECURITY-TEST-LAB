"""L2 · OllamaAdapter — HTTP calls to the local Ollama daemon.

Uses POST {host}/api/chat so the system prompt (defense) and user prompt
(attack) go in as separate messages — which is exactly the boundary we test.
Uses the stdlib (urllib) to avoid extra dependencies.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request

from .base import BaseModelAdapter, GenerationResult


class OllamaAdapter(BaseModelAdapter):
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host.rstrip("/")
        # Seconds per call. Long generations (the AI summary report) raise it.
        self.timeout = 200

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
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            text = data.get("message", {}).get("content", "")
            return GenerationResult(text, self.model_name, (time.time() - t0) * 1000, options)
        except Exception as e:  # noqa: BLE001 - capture any runtime error safely
            return GenerationResult("", self.model_name, (time.time() - t0) * 1000, options, str(e))

    def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        options: dict | None = None,
    ) -> GenerationResult:
        """Multi-turn chat. `messages` is a list of {'role','content'} dicts
        already in send-order (oldest first, latest user message last).
        The system prompt is prepended if given.
        """
        options = options or {}
        full = []
        if system_prompt:
            full.append({"role": "system", "content": system_prompt})
        full.extend(messages)
        body = json.dumps({
            "model": self.model_name,
            "messages": full,
            "stream": False,
            "options": options,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read())
            text = data.get("message", {}).get("content", "")
            return GenerationResult(text, self.model_name, (time.time() - t0) * 1000, options)
        except Exception as e:  # noqa: BLE001
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

    # Model name Ollama accepts: lowercase letters/digits and . _ -, with an
    # optional :tag. Kept strict so a name typed in the UI can't become a shell
    # surprise or a malformed registry ref.
    _NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}(:[a-z0-9._-]{1,32})?")

    @classmethod
    def create_from_gguf(cls, name: str, gguf_path: str) -> None:
        """Register a local .gguf file into Ollama as a usable model.

        Uses the `ollama create` CLI rather than the HTTP /api/create endpoint:
        recent Ollama versions moved that endpoint to a blob-digest upload flow,
        and the CLI hides those differences and works across versions. Ollama
        copies the file into its own model store, so the GGUF is duplicated on
        disk — the caller's UI warns about the extra space.

        Raises ValueError with a user-fixable message on any problem.
        """
        name = (name or "").strip()
        if not cls._NAME_RE.fullmatch(name):
            raise ValueError(
                "Geçersiz model adı. Sadece küçük harf, rakam ve . _ - kullan "
                "(örn. benim-modelim veya benim-modelim:v1)."
            )
        path = os.path.abspath(os.path.expanduser((gguf_path or "").strip().strip('"')))
        if not os.path.isfile(path):
            raise ValueError(f"Dosya bulunamadı: {path}")
        if not path.lower().endswith(".gguf"):
            raise ValueError("Dosya .gguf uzantılı olmalı.")
        if shutil.which("ollama") is None:
            raise ValueError(
                "'ollama' komutu bulunamadı. Ollama kurulu ve PATH'te olmalı."
            )
        modelfile: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".Modelfile", delete=False, encoding="utf-8"
            ) as mf:
                mf.write(f"FROM {path}\n")
                modelfile = mf.name
            proc = subprocess.run(
                ["ollama", "create", name, "-f", modelfile],
                capture_output=True, text=True, timeout=900,
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise ValueError(
                    "Ollama modeli oluşturamadı: " + (detail or "bilinmeyen hata")
                )
        except subprocess.TimeoutExpired:
            raise ValueError(
                "Ollama create zaman aşımına uğradı (15 dk). Dosya çok büyük "
                "olabilir; terminalden `ollama create` ile deneyebilirsin."
            ) from None
        finally:
            if modelfile:
                try:
                    os.unlink(modelfile)
                except OSError:
                    pass
