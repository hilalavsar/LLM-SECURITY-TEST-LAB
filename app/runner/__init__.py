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


def list_all_cases() -> dict[str, list]:
    """Return all cases grouped by language: {'tr': [...], 'en': [...]}."""
    return {lang: load_test_suite(path).cases for lang, path in CORPUS_PATHS.items()}


def get_case(lang: str, case_id: str):
    """Fetch a single case by language and id, or None."""
    if lang not in CORPUS_PATHS:
        return None
    for case in load_test_suite(CORPUS_PATHS[lang]).cases:
        if case.id == case_id:
            return case
    return None


def run_single_attack(lang: str, case_id: str, model: str, config_name: str) -> dict:
    """Execute one attack and return {response, verdict, reason, latency_ms, error}.
    Not persisted to DB — this is for interactive corpus browsing."""
    case = get_case(lang, case_id)
    if case is None:
        return {"error": f"Case not found: {lang}/{case_id}"}
    cfg = load_target_system(DEFENSES_DIR / f"{config_name}.yaml")
    adapter = OllamaAdapter(model, Settings.OLLAMA_HOST)
    res = adapter.generate(case.prompt, cfg.system_prompt, Settings.GEN_OPTIONS)
    verdict, reason = evaluate_case(case, res.text, cfg.canary)
    return {
        "response": res.text,
        "verdict": verdict.value,
        "reason": reason,
        "latency_ms": round(res.latency_ms),
        "error": res.error,
    }


def get_run(run_id: str) -> dict | None:
    with _LOCK:
        return _RUNS.get(run_id)


def start_run(model: str, config_names: list[str], languages: list[str] | None = None,
              judge_model: str = "") -> str:
    """Kick off one or more test runs (one per language) sequentially in a thread.

    Returns the FIRST run_id; a chain-run_id for the second language is created
    inside the worker so the browser can follow both by polling the first.

    judge_model="" disables the judge for this run; otherwise the given Ollama
    model is used as the semantic judge for PENDING cases.
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
                         "languages": languages, "child_run_ids": [],
                         "judge_model": judge_model}
    threading.Thread(target=_execute_multi,
                     args=(run_id, ts, model, config_names, languages, suites, judge_model),
                     daemon=True).start()
    return run_id


def _execute_multi(run_id: str, ts: str, model: str, config_names: list[str],
                   languages: list[str], suites: dict, judge_model: str) -> None:
    """Run each language as its own DB Run, sharing one progress counter."""
    for i, lang in enumerate(languages):
        child_id = run_id if i == 0 else f"{ts}-{lang}"
        if i > 0:
            with _LOCK:
                _RUNS[run_id]["child_run_ids"].append(child_id)
        _execute_single(run_id, child_id, model, config_names, suites[lang], judge_model)
    with _LOCK:
        _RUNS[run_id]["status"] = "done"


def _execute_single(progress_id: str, save_id: str, model: str,
                    config_names: list[str], suite, judge_model: str) -> None:
    adapter = OllamaAdapter(model, Settings.OLLAMA_HOST)
    # Judge is opt-in per run; empty judge_model disables the semantic layer.
    judge_adapter = (
        OllamaAdapter(judge_model, Settings.OLLAMA_HOST) if judge_model else None
    )
    rows: list[Result] = []
    for cname in config_names:
        cfg = load_target_system(DEFENSES_DIR / f"{cname}.yaml")
        for case in suite.cases:
            res = adapter.generate(case.prompt, cfg.system_prompt, Settings.GEN_OPTIONS)
            verdict, reason = evaluate_case(
                case, res.text, cfg.canary, judge_adapter=judge_adapter
            )
            rows.append(Result(
                config=cname, case_id=case.id, category=case.category.value,
                owasp=case.owasp.value, detection=case.detection_method.value,
                verdict=verdict.value, reason=reason, prompt=case.prompt,
                response=res.text, latency_ms=round(res.latency_ms), error=res.error))
            with _LOCK:
                _RUNS[progress_id]["done"] += 1

    with SessionLocal() as s:
        run = Run(id=save_id, model=model, configs=",".join(config_names),
                  judge_model=judge_model)
        run.results = rows
        s.add(run)
        s.commit()


# --- read helpers used by the views ---------------------------------------


def _row_to_dict(r: Result) -> dict:
    return {"pk": r.pk, "config": r.config, "id": r.case_id, "category": r.category,
            "owasp": r.owasp, "detection": r.detection, "verdict": r.verdict,
            "reason": r.reason, "prompt": r.prompt, "response": r.response,
            "latency_ms": r.latency_ms, "error": r.error}


def finalize_result(pk: int, verdict: str) -> dict | None:
    """Manually override a result's verdict (used for PENDING cases)."""
    if verdict not in ("PASS", "FAIL", "ERROR"):
        return None
    with SessionLocal() as s:
        result = s.get(Result, pk)
        if not result:
            return None
        prev = result.verdict
        result.verdict = verdict
        result.reason = f"[manual] finalized from {prev} by user"
        s.commit()
        return {"pk": pk, "verdict": verdict, "run_id": result.run_id}


def re_evaluate_run(source_run_id: str, judge_model: str) -> str | None:
    """Re-score an existing run's responses with a (possibly different) judge.

    Model responses are NOT re-generated — evaluator (rule-based + judge) is
    re-applied on the stored responses and a NEW Run row is saved. The source
    run is left untouched. Returns the new run_id, or None on error.
    """
    with SessionLocal() as s:
        source = s.get(Run, source_run_id)
        if not source:
            return None
        lang = _lang_of(source)
        if lang not in CORPUS_PATHS:
            return None

        judge_adapter = (
            OllamaAdapter(judge_model, Settings.OLLAMA_HOST) if judge_model else None
        )
        # Cache loaded configs and cases by id for the pass.
        cfg_cache: dict[str, object] = {}
        cases_by_id = {c.id: c for c in load_test_suite(CORPUS_PATHS[lang]).cases}

        ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        new_id = f"{ts}-{lang}"
        new_rows: list[Result] = []
        for r in source.results:
            if r.config not in cfg_cache:
                cfg_cache[r.config] = load_target_system(
                    DEFENSES_DIR / f"{r.config}.yaml"
                )
            cfg = cfg_cache[r.config]
            case = cases_by_id.get(r.case_id)
            if case is None:
                # Case removed/renamed since the original run — keep old verdict.
                verdict = r.verdict
                reason = f"{r.reason} (case not in current corpus)"
            else:
                v, reason = evaluate_case(
                    case, r.response, cfg.canary, judge_adapter=judge_adapter
                )
                verdict = v.value
            new_rows.append(Result(
                config=r.config, case_id=r.case_id, category=r.category,
                owasp=r.owasp, detection=r.detection, verdict=verdict, reason=reason,
                prompt=r.prompt, response=r.response,
                latency_ms=r.latency_ms, error=r.error,
            ))

        new_run = Run(id=new_id, model=source.model, configs=source.configs,
                      judge_model=judge_model)
        new_run.results = new_rows
        s.add(new_run)
        s.commit()
        return new_id


def _summarize(rows: list[dict], config_names: list[str]) -> dict:
    """ASR per config over auto-decided (PASS/FAIL) cases only."""
    summary = {}
    for c in config_names:
        crows = [r for r in rows if r["config"] == c]
        decided = [r for r in crows if r["verdict"] in ("PASS", "FAIL")]
        fails = [r for r in crows if r["verdict"] == "FAIL"]
        pending = [r for r in crows if r["verdict"] == "PENDING"]
        asr = round(100 * len(fails) / len(decided), 1) if decided else 0.0
        summary[c] = {"asr": asr, "fail": len(fails), "decided": len(decided),
                      "pending": len(pending), "total": len(crows)}
    return summary


def _judge_stats(rows: list[dict]) -> dict:
    """Break down how each result got its verdict — for the dashboard 'judge impact' box.

    We infer the source from `reason` prefix set by the evaluator/judge modules.
    """
    stats = {"judge_pass": 0, "judge_fail": 0, "judge_unparseable": 0,
             "judge_error": 0, "manual": 0, "rule_pass": 0, "rule_fail": 0,
             "rule_pending_no_judge": 0}
    for r in rows:
        reason = r.get("reason") or ""
        verdict = r.get("verdict")
        if reason.startswith("[judge]"):
            if verdict == "PASS":
                stats["judge_pass"] += 1
            elif verdict == "FAIL":
                stats["judge_fail"] += 1
        elif reason.startswith("Judge verdict unparseable"):
            stats["judge_unparseable"] += 1
        elif reason.startswith("Judge model error"):
            stats["judge_error"] += 1
        elif reason.startswith("[manual]"):
            stats["manual"] += 1
        elif reason == "Needs semantic judge":
            stats["rule_pending_no_judge"] += 1
        elif verdict == "FAIL":
            stats["rule_fail"] += 1
        elif verdict == "PASS":
            stats["rule_pass"] += 1
    stats["judge_decided"] = stats["judge_pass"] + stats["judge_fail"]
    stats["judge_touched"] = (stats["judge_decided"] + stats["judge_unparseable"]
                              + stats["judge_error"])
    return stats


def load_run(run_id: str) -> dict | None:
    with SessionLocal() as s:
        run = s.get(Run, run_id)
        if not run:
            return None
        configs = run.configs.split(",")
        results = [_row_to_dict(r) for r in run.results]
        return {"id": run.id, "model": run.model, "configs": configs,
                "judge_model": run.judge_model or "",
                "results": results, "summary": _summarize(results, configs),
                "judge_stats": _judge_stats(results)}


def list_runs() -> list[dict]:
    with SessionLocal() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        return [{"id": r.id, "model": r.model, "configs": r.configs.split(","),
                 "judge_model": r.judge_model or ""}
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


def judge_impact_pairs(rows: list[dict]) -> list[dict]:
    """For each (model, lang) that has BOTH judge-off and judge-on rows,
    return a pair with per-config deltas (with_judge - no_judge).

    Since `rows` is already deduped by (model, lang, judge_model) and sorted
    with empty judge_model first, we can walk once and group.
    """
    grouped: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["model"], r["lang"])
        state = "no_judge" if not r["judge"] else "with_judge"
        # Keep the first occurrence per state (which is the latest, given the sort order).
        grouped.setdefault(key, {"no_judge": None, "with_judge": None})
        if grouped[key][state] is None:
            grouped[key][state] = r

    pairs = []
    for (model, lang), g in grouped.items():
        n = g["no_judge"]
        w = g["with_judge"]
        if n is None or w is None:
            continue
        cells = []
        for cfg in sorted(set(n["configs"]) | set(w["configs"])):
            no_val = n["asr"].get(cfg)
            with_val = w["asr"].get(cfg)
            delta = (round(with_val - no_val, 1)
                     if no_val is not None and with_val is not None else None)
            cells.append({
                "config": cfg,
                "no_judge": no_val,
                "with_judge": with_val,
                "delta": delta,
                "no_judge_pending": n["pending"].get(cfg, 0),
                "with_judge_pending": w["pending"].get(cfg, 0),
            })
        pairs.append({
            "model": model,
            "lang": lang,
            "no_judge_run": n["run_id"],
            "with_judge_run": w["run_id"],
            "judge_model": w["judge"],
            "cells": cells,
        })
    return sorted(pairs, key=lambda p: (p["model"], p["lang"]))


def model_comparison() -> tuple[list[str], list[dict]]:
    """Latest run per (model, language, judge_model) — one row each.

    Returns (configs, rows) where rows is a flat list of dicts with keys:
    model, lang, judge, run_id, configs, asr {cfg: pct}, pending {cfg: n}.
    """
    with SessionLocal() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        latest: dict[tuple, Run] = {}
        for r in runs:
            key = (r.model, _lang_of(r), r.judge_model or "")
            latest.setdefault(key, r)

        configs_seen: list[str] = []
        rows: list[dict] = []
        for (model, lang, judge), run in latest.items():
            result_rows = [_row_to_dict(x) for x in run.results]
            cfgs = run.configs.split(",")
            summ = _summarize(result_rows, cfgs)
            for c in cfgs:
                if c not in configs_seen:
                    configs_seen.append(c)
            rows.append({
                "model": model,
                "lang": lang,
                "judge": judge,
                "run_id": run.id,
                "configs": cfgs,
                "asr": {c: summ[c]["asr"] for c in cfgs},
                "pending": {c: summ[c]["pending"] for c in cfgs},
            })
        # Stable sort: model, then lang, then judge (empty first as "no judge").
        rows.sort(key=lambda r: (r["model"], r["lang"], r["judge"]))
        return sorted(configs_seen), rows
