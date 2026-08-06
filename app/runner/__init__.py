"""L4 · Orchestration — test runner (database-backed).

Runs the attack corpus against one or more defense configs on a chosen model,
evaluates each response, and stores the run in PostgreSQL. Runs happen in a
background thread so the web UI stays responsive; progress is tracked in an
in-memory dict and polled by the browser.
"""

from __future__ import annotations

import threading
import time

from app.adapters.ollama import OllamaAdapter
from app.config import CORPUS_PATHS, DEFENSES_DIR, Settings
from app.db import SessionLocal
from app.evaluator.rules import evaluate_case
from app.models import Result, Run
from app.schemas.loader import load_target_system, load_test_suite

_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()


def list_defense_configs() -> list[str]:
    return sorted(p.stem for p in DEFENSES_DIR.glob("*.yaml"))


def get_run(run_id: str) -> dict | None:
    with _LOCK:
        return _RUNS.get(run_id)


def start_run(model: str, config_names: list[str], languages: list[str] | None = None) -> str:
    """Kick off one or more test runs (one per language) sequentially in a thread.

    Returns the FIRST run_id; a chain-run_id for the second language is created
    inside the worker so the browser can follow both by polling the first.
    """
    languages = languages or ["tr"]
    # Millisecond suffix prevents collisions on rapid double-submit.
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    run_id = f"{ts}-{languages[0]}"
    suites = {lang: load_test_suite(CORPUS_PATHS[lang]) for lang in languages}
    total = sum(len(config_names) * len(s.cases) for s in suites.values())
    with _LOCK:
        _RUNS[run_id] = {"id": run_id, "model": model, "status": "running",
                         "done": 0, "total": total, "configs": config_names,
                         "languages": languages, "child_run_ids": []}
    threading.Thread(target=_execute_multi,
                     args=(run_id, ts, model, config_names, languages, suites),
                     daemon=True).start()
    return run_id


def _execute_multi(run_id: str, ts: str, model: str, config_names: list[str],
                   languages: list[str], suites: dict) -> None:
    """Run each language as its own DB Run, sharing one progress counter."""
    for i, lang in enumerate(languages):
        child_id = run_id if i == 0 else f"{ts}-{lang}"
        if i > 0:
            with _LOCK:
                _RUNS[run_id]["child_run_ids"].append(child_id)
        _execute_single(run_id, child_id, model, config_names, suites[lang])
    with _LOCK:
        _RUNS[run_id]["status"] = "done"


def _execute_single(progress_id: str, save_id: str, model: str,
                    config_names: list[str], suite) -> None:
    adapter = OllamaAdapter(model, Settings.OLLAMA_HOST)
    rows: list[Result] = []
    for cname in config_names:
        cfg = load_target_system(DEFENSES_DIR / f"{cname}.yaml")
        for case in suite.cases:
            res = adapter.generate(case.prompt, cfg.system_prompt, Settings.GEN_OPTIONS)
            verdict, reason = evaluate_case(case, res.text, cfg.canary)
            rows.append(Result(
                config=cname, case_id=case.id, category=case.category.value,
                owasp=case.owasp.value, detection=case.detection_method.value,
                verdict=verdict.value, reason=reason, prompt=case.prompt,
                response=res.text, latency_ms=round(res.latency_ms), error=res.error))
            with _LOCK:
                _RUNS[progress_id]["done"] += 1

    with SessionLocal() as s:
        run = Run(id=save_id, model=model, configs=",".join(config_names))
        run.results = rows
        s.add(run)
        s.commit()


# --- read helpers used by the views ---------------------------------------


def _row_to_dict(r: Result) -> dict:
    return {"config": r.config, "id": r.case_id, "category": r.category,
            "owasp": r.owasp, "detection": r.detection, "verdict": r.verdict,
            "reason": r.reason, "prompt": r.prompt, "response": r.response,
            "latency_ms": r.latency_ms, "error": r.error}


def _summarize(rows: list[dict], config_names: list[str]) -> dict:
    """ASR per config over auto-decided (PASS/FAIL) cases only."""
    summary = {}
    for c in config_names:
        crows = [r for r in rows if r["config"] == c]
        decided = [r for r in crows if r["verdict"] in ("PASS", "FAIL")]
        fails = [r for r in crows if r["verdict"] == "FAIL"]
        review = [r for r in crows if r["verdict"] == "REVIEW"]
        asr = round(100 * len(fails) / len(decided), 1) if decided else 0.0
        summary[c] = {"asr": asr, "fail": len(fails), "decided": len(decided),
                      "review": len(review), "total": len(crows)}
    return summary


def load_run(run_id: str) -> dict | None:
    with SessionLocal() as s:
        run = s.get(Run, run_id)
        if not run:
            return None
        configs = run.configs.split(",")
        results = [_row_to_dict(r) for r in run.results]
        return {"id": run.id, "model": run.model, "configs": configs,
                "results": results, "summary": _summarize(results, configs)}


def list_runs() -> list[dict]:
    with SessionLocal() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        return [{"id": r.id, "model": r.model, "configs": r.configs.split(",")}
                for r in runs]


def _lang_of(run: "Run") -> str:
    """Derive language from run_id suffix (e.g. '-tr' / '-en'); legacy runs default to 'tr'."""
    for lang in ("tr", "en"):
        if run.id.endswith(f"-{lang}"):
            return lang
    # Fallback: sniff from a case_id prefix.
    if run.results:
        cid = run.results[0].case_id
        if cid.startswith("EN-"):
            return "en"
    return "tr"


def model_comparison() -> tuple[list[str], list[str], dict]:
    """Latest run per (model, language) -> ASR per config, for the comparison table.

    Returns (configs, languages, data) where data[model][lang] = {run_id, asr:{cfg:pct}}.
    """
    with SessionLocal() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        latest: dict[tuple[str, str], Run] = {}
        for r in runs:
            latest.setdefault((r.model, _lang_of(r)), r)
        configs: list[str] = []
        languages: list[str] = []
        data: dict[str, dict] = {}
        for (model, lang), run in latest.items():
            rows = [_row_to_dict(x) for x in run.results]
            cfgs = run.configs.split(",")
            summ = _summarize(rows, cfgs)
            data.setdefault(model, {})
            data[model][lang] = {"run_id": run.id,
                                 "asr": {c: summ[c]["asr"] for c in cfgs}}
            for c in cfgs:
                if c not in configs:
                    configs.append(c)
            if lang not in languages:
                languages.append(lang)
        return sorted(configs), sorted(languages), data
