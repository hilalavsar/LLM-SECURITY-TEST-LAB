"""Run the full test matrix across all 3 models, both languages, back-to-back.

Matrix:
  3 models × 2 languages × 3 configs × 9 cases = 162 model calls total
  Estimated runtime: 15-40 minutes depending on CPU offload

Each (model, language) pair is saved as its own Run row in PostgreSQL,
so the /compare page will show 6 rows (3 models × 2 languages) when done.

Prerequisites:
  - Ollama daemon running (http://localhost:11434)
  - All 3 models pulled: qwen2.5:7b, llama3.1:8b-instruct-q4_K_M, mistral:7b-instruct-q4_K_M
  - PostgreSQL container up (docker compose up -d)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make `app` importable no matter which directory we launch from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.adapters.ollama import OllamaAdapter
from app.config import CORPUS_PATHS, DEFENSES_DIR, Settings
from app.db import SessionLocal, init_db
from app.evaluator.rules import evaluate_case
from app.models import Result, Run
from app.schemas.loader import load_target_system, load_test_suite

MODELS = [
    "llama3.1:8b-instruct-q4_K_M",   # fastest first, keeps momentum
    "qwen2.5:7b",
    "mistral:7b-instruct-q4_K_M",
]
LANGUAGES = ["tr", "en"]
CONFIGS = ["config0_none", "config1_basic", "config2_hardened"]


def preflight() -> None:
    """Fail early with a clear message if the environment is not ready."""
    # DB reachable?
    try:
        init_db()
    except Exception as e:
        sys.exit(f"[FATAL] PostgreSQL is not reachable: {e}\n"
                 f"        Run: docker compose up -d")

    # Ollama reachable and models present?
    available = set(OllamaAdapter.list_models(Settings.OLLAMA_HOST))
    if not available:
        sys.exit(f"[FATAL] Ollama is not responding at {Settings.OLLAMA_HOST}\n"
                 f"        Run: ollama serve  (or open Ollama desktop app)")
    missing = [m for m in MODELS if m not in available]
    if missing:
        sys.exit(f"[FATAL] Missing models: {missing}\n"
                 f"        Pull with:  ollama pull <model>")

    # Corpus files load cleanly?
    for lang in LANGUAGES:
        suite = load_test_suite(CORPUS_PATHS[lang])
        print(f"  [OK] {lang.upper()} corpus loaded ({len(suite.cases)} cases)")


def run_one(model: str, lang: str, suite, defenses: dict) -> None:
    """Run one (model, language) pair and persist as a single Run row."""
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    run_id = f"{ts}-{lang}"
    print(f"\n>>> {model} × {lang.upper()}   run_id={run_id}")

    adapter = OllamaAdapter(model, Settings.OLLAMA_HOST)
    rows: list[Result] = []
    t0 = time.time()

    for cname in CONFIGS:
        cfg = defenses[cname]
        for case in suite.cases:
            res = adapter.generate(case.prompt, cfg.system_prompt, Settings.GEN_OPTIONS)
            verdict, reason = evaluate_case(case, res.text, cfg.canary)
            rows.append(Result(
                config=cname, case_id=case.id, category=case.category.value,
                owasp=case.owasp.value, detection=case.detection_method.value,
                verdict=verdict.value, reason=reason, prompt=case.prompt,
                response=res.text, latency_ms=round(res.latency_ms), error=res.error))
            marker = "❌" if verdict.value == "FAIL" else ("⚠ " if verdict.value == "REVIEW" else "✓ ")
            print(f"    {marker} [{cname}] {case.id} -> {verdict.value}")

    with SessionLocal() as s:
        run = Run(id=run_id, model=model, configs=",".join(CONFIGS))
        run.results = rows
        s.add(run)
        s.commit()
    print(f"    ✓ Saved in {round(time.time() - t0)} sec")


def main() -> None:
    print("=" * 64)
    print("LLM Security Test Lab — Full Model x Language Matrix Run")
    print("=" * 64)
    print("Preflight checks...")
    preflight()

    total_runs = len(MODELS) * len(LANGUAGES)
    total_calls = total_runs * len(CONFIGS) * 9  # 9 cases per corpus
    print(f"\nPlanned: {total_runs} runs, {total_calls} model calls total")
    print(f"Models:     {MODELS}")
    print(f"Languages:  {LANGUAGES}")
    print(f"Configs:    {CONFIGS}")

    suites = {lang: load_test_suite(CORPUS_PATHS[lang]) for lang in LANGUAGES}
    defenses = {c: load_target_system(DEFENSES_DIR / f"{c}.yaml") for c in CONFIGS}

    overall_start = time.time()
    completed = 0
    for model in MODELS:
        print(f"\n{'=' * 64}\n=== MODEL {completed // len(LANGUAGES) + 1}/{len(MODELS)}: {model}\n{'=' * 64}")
        for lang in LANGUAGES:
            try:
                run_one(model, lang, suites[lang], defenses)
                completed += 1
                elapsed_min = (time.time() - overall_start) / 60
                print(f"    [progress: {completed}/{total_runs} runs done, {elapsed_min:.1f} min elapsed]")
            except Exception as e:
                print(f"    [ERROR] {model} × {lang} failed: {e}")
                print("    continuing with next combination...")
            # Small gap so run_ids never collide even on very fast runs
            time.sleep(1)

    total_min = (time.time() - overall_start) / 60
    print(f"\n{'=' * 64}")
    print(f"ALL DONE — {completed}/{total_runs} runs saved in {total_min:.1f} min")
    print(f"View results at: http://127.0.0.1:5000/compare")
    print("=" * 64)


if __name__ == "__main__":
    main()
