"""Run the test matrix for several models back-to-back from the command line.

Uses the same runner as the web UI (app.runner.start_run), so each model is
saved as one Run with the judge, dataset and defense-text snapshot exactly as
if it had been started from the "Çalıştır" page. Results appear on /compare.

Defaults: 3 local models x 4 built-in defense levels x 52 scenarios, judged by
Settings.JUDGE_MODEL.

Examples:
  python scripts/run_all_models.py
  python scripts/run_all_models.py --models qwen2.5:7b mistral:7b-instruct-q4_K_M
  python scripts/run_all_models.py --models gemini:gemini-2.5-flash --no-judge
  python scripts/run_all_models.py --configs config0_none_en config3_maximal_en

Prerequisites:
  - PostgreSQL container up (docker compose up -d)
  - Ollama running with the target and judge models pulled
  - GEMINI_API_KEY in .env for gemini:<model> targets
API providers added from the Sağlayıcılar page live in the web app's memory
only, so api:<provider>:<model> targets can't be run from here.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `app` importable no matter which directory we launch from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from app import datasets, defenses, runner  # noqa: E402
from app.adapters.ollama import OllamaAdapter  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import init_db  # noqa: E402

DEFAULT_MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b",
    "mistral:7b-instruct-q4_K_M",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="target models (Ollama names, or gemini:<model>)")
    p.add_argument("--configs", nargs="+", default=list(defenses.BUILTIN),
                   help="defense layers (default: the 4 built-in levels)")
    p.add_argument("--judge", default=Settings.JUDGE_MODEL,
                   help="judge model for cases the rules can't decide")
    p.add_argument("--no-judge", action="store_true",
                   help="rules only; undecided cases stay PENDING")
    p.add_argument("--dataset", default="",
                   help="user dataset id from the Datasetler page (default: built-in corpus)")
    return p.parse_args()


def _ollama_has(name: str, available: set[str]) -> bool:
    return name in available or f"{name}:latest" in available


def preflight(models: list[str], configs: list[str], judge: str, dataset: str) -> None:
    """Fail early with a clear message if the environment is not ready."""
    try:
        init_db()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[FATAL] PostgreSQL is not reachable: {e}\n"
                 f"        Run: docker compose up -d")

    known = {layer["name"] for layer in defenses.list_layers()}
    unknown = [c for c in configs if c not in known]
    if unknown:
        sys.exit(f"[FATAL] Unknown defense layers: {unknown}\n"
                 f"        Available: {sorted(known)}")

    if not datasets.exists(dataset):
        sys.exit(f"[FATAL] Dataset not found: {dataset}")

    if any(m.startswith("api:") for m in models + [judge]):
        sys.exit("[FATAL] api:<provider>:<model> only works from the web UI "
                 "(providers are kept in the app's memory).")
    if any(m.startswith("gemini:") for m in models + [judge]) and not Settings.GEMINI_API_KEY:
        sys.exit("[FATAL] gemini:<model> needs GEMINI_API_KEY in .env")

    local = [m for m in models + [judge] if m and not m.startswith("gemini:")]
    if local:
        available = set(OllamaAdapter.list_models(Settings.OLLAMA_HOST))
        if not available:
            sys.exit(f"[FATAL] Ollama is not responding at {Settings.OLLAMA_HOST}\n"
                     f"        Run: ollama serve  (or open the Ollama app)")
        missing = [m for m in local if not _ollama_has(m, available)]
        if missing:
            sys.exit(f"[FATAL] Missing Ollama models: {missing}\n"
                     f"        Pull with:  ollama pull <model>")


def run_one(model: str, configs: list[str], judge: str, dataset: str) -> str | None:
    """Start one run through the web runner and wait for it. Returns run_id."""
    run_id = runner.start_run(model, configs, ["en"], judge_model=judge, dataset=dataset)
    print(f"\n>>> {model}   run_id={run_id}")
    last = -1
    while True:
        state = runner.get_run(run_id)
        if state["done"] != last:
            last = state["done"]
            print(f"    {last}/{state['total']}", end="\r", flush=True)
        if state["status"] != "running":
            break
        time.sleep(2)
    print()
    if state["status"] == "error":
        print(f"    [ERROR] {state.get('error')}")
        return None

    results = runner.load_run(run_id)["results"]
    counts = {v: sum(r["verdict"] == v for r in results)
              for v in ("PASS", "FAIL", "PENDING", "ERROR")}
    decided = counts["PASS"] + counts["FAIL"]
    asr = f"{100 * counts['FAIL'] / decided:.1f}%" if decided else "-"
    print(f"    PASS {counts['PASS']}  FAIL {counts['FAIL']}  PENDING {counts['PENDING']}"
          f"  ERROR {counts['ERROR']}  ->  ASR {asr}")
    return run_id


def main() -> None:
    args = parse_args()
    judge = "" if args.no_judge else args.judge
    print("LLM Security Test Lab - batch run")
    print(f"  models : {args.models}")
    print(f"  configs: {args.configs}")
    print(f"  judge  : {judge or 'off'}")
    print(f"  dataset: {datasets.label(args.dataset)}")
    preflight(args.models, args.configs, judge, args.dataset)

    cases = len(datasets.load_suite(args.dataset).cases)
    print(f"\nPlanned: {len(args.models)} runs x {len(args.configs)} configs x "
          f"{cases} cases = {len(args.models) * len(args.configs) * cases} model calls")

    started = time.time()
    saved = []
    for model in args.models:
        try:
            run_id = run_one(model, args.configs, judge, args.dataset)
        except Exception as e:  # noqa: BLE001
            print(f"    [ERROR] {model}: {e}")
            run_id = None
        if run_id:
            saved.append(run_id)
        time.sleep(1)  # run ids carry a timestamp; keep them distinct

    print(f"\nDone: {len(saved)}/{len(args.models)} runs saved in "
          f"{(time.time() - started) / 60:.1f} min")
    print("View results at: http://127.0.0.1:5000/compare")


if __name__ == "__main__":
    main()
