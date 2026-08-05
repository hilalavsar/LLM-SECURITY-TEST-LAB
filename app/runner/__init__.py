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
from app.config import CORPUS_PATH, DEFENSES_DIR, Settings
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


def start_run(model: str, config_names: list[str]) -> str:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    suite = load_test_suite(CORPUS_PATH)
    total = len(config_names) * len(suite.cases)
    with _LOCK:
        _RUNS[run_id] = {"id": run_id, "model": model, "status": "running",
                         "done": 0, "total": total, "configs": config_names}
    threading.Thread(target=_execute, args=(run_id, model, config_names, suite),
                     daemon=True).start()
    return run_id


def _execute(run_id: str, model: str, config_names: list[str], suite) -> None:
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
                _RUNS[run_id]["done"] += 1

    with SessionLocal() as s:
        run = Run(id=run_id, model=model, configs=",".join(config_names))
        run.results = rows
        s.add(run)
        s.commit()
    with _LOCK:
        _RUNS[run_id]["status"] = "done"


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


def model_comparison() -> tuple[list[str], dict]:
    """Latest run per model -> ASR per config, for one comparison table."""
    with SessionLocal() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        latest: dict[str, Run] = {}
        for r in runs:
            latest.setdefault(r.model, r)
        configs: list[str] = []
        data: dict[str, dict] = {}
        for model, run in latest.items():
            rows = [_row_to_dict(x) for x in run.results]
            cfgs = run.configs.split(",")
            summ = _summarize(rows, cfgs)
            data[model] = {"run_id": run.id, "asr": {c: summ[c]["asr"] for c in cfgs}}
            for c in cfgs:
                if c not in configs:
                    configs.append(c)
        return sorted(configs), data
