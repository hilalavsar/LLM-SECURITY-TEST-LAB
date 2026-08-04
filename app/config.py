"""Konfigurasyon — .env dosyasindan ayarlari okur.

Pydantic-settings ile tip guvenli. Hafta 2'de genisletilecek.
"""

from __future__ import annotations
import os


class Settings:
    """Basit baslangic; sonra pydantic-settings'e tasinacak."""
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    TARGET_MODEL_PRIMARY = os.getenv("TARGET_MODEL_PRIMARY", "qwen2.5:7b")
    TARGET_MODEL_SECONDARY = os.getenv(
        "TARGET_MODEL_SECONDARY", "llama3.1:8b-instruct-q4_K_M"
    )
    DATABASE_URL = os.getenv("DATABASE_URL", "")
