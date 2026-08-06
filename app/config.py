"""Configuration — reads settings from environment variables.

Kept intentionally small for now; will move to pydantic-settings in Week 2.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: .../LLM-SECURITY-TEST-LAB
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFENSES_DIR = DATA_DIR / "defenses"
# Language-keyed corpus registry — used by runner to pick the right suite.
CORPUS_PATHS = {
    "tr": DATA_DIR / "test_cases" / "corpus_tr_v0.yaml",
    "en": DATA_DIR / "test_cases" / "corpus_en_v0.yaml",
}
# Kept for backward compatibility with legacy imports.
CORPUS_PATH = CORPUS_PATHS["tr"]


class Settings:
    """Simple starter settings; env-overridable."""

    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    TARGET_MODEL_PRIMARY = os.getenv("TARGET_MODEL_PRIMARY", "qwen2.5:7b")
    TARGET_MODEL_SECONDARY = os.getenv(
        "TARGET_MODEL_SECONDARY", "llama3.1:8b-instruct-q4_K_M"
    )
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://labuser:labpass@localhost:5432/llmseclab",
    )

    # Generation defaults — fixed for reproducibility (system-prompt study).
    GEN_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 320}
