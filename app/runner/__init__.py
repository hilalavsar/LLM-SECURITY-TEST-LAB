"""L4 · Orchestration — test runner (database-backed).

Runs the attack corpus against one or more defense configs on a chosen model,
evaluates each response, and stores the run in PostgreSQL. Runs happen in a
background thread so the web UI stays responsive; progress is tracked in an
in-memory dict and polled by the browser.
"""

from __future__ import annotations

import json
import threading
import time

from app import datasets, defenses, providers
from app.adapters.ollama import OllamaAdapter
from app.config import CORPUS_PATHS, DEFENSES_DIR, Settings
from app.db import SessionLocal
from app.evaluator.rules import Verdict, evaluate_case
from app.models import Result, Run
from app.schemas.loader import load_target_system, load_test_suite
from app.schemas.test_case import CANARY

# Gemini SDK is optional — only imported/used when a Gemini-tagged model is
# selected. Keeping this at module load time so the factory below can branch.
try:
    from app.adapters.gemini import KNOWN_GEMINI_MODELS, GeminiAdapter
    _GEMINI_OK = True
except ImportError:
    KNOWN_GEMINI_MODELS = []
    GeminiAdapter = None  # type: ignore[assignment]
    _GEMINI_OK = False


def _make_adapter(model_name: str):
    """Route model calls to the right runtime:
      'api:<provider>:<model>'  -> OpenAI-compatible endpoint added from the UI
      'gemini:<model>'          -> Gemini SDK (key from .env)
      anything else             -> local Ollama
    Bare 'gemini-1.5-pro' is still accepted for older stored runs.
    """
    if model_name.startswith(providers.PREFIX):
        return providers.build_adapter(model_name)
    if model_name.startswith("gemini:"):
        bare = model_name[len("gemini:"):]
        return GeminiAdapter(bare, Settings.GEMINI_API_KEY)
    if _GEMINI_OK and model_name in KNOWN_GEMINI_MODELS:
        return GeminiAdapter(model_name, Settings.GEMINI_API_KEY)
    return OllamaAdapter(model_name, Settings.OLLAMA_HOST)


def check_models(*model_names: str) -> None:
    """Resolve API-provider models up front, so a provider missing from this
    session fails in the request instead of inside the background thread.
    Builds adapters only; makes no network calls."""
    for name in model_names:
        if name and name.startswith(providers.PREFIX):
            providers.build_adapter(name)

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


def run_single_attack(lang: str, case_id: str, model: str, config_name: str,
                      judge_model: str = "") -> dict:
    """Execute one corpus attack (not persisted). Optional judge for semantic cases."""
    case = get_case(lang, case_id)
    if case is None:
        return {"error": f"Case not found: {lang}/{case_id}"}
    cfg = load_target_system(DEFENSES_DIR / f"{config_name}.yaml")
    adapter = _make_adapter(model)
    judge_adapter = (
        _make_adapter(judge_model) if judge_model else None
    )
    res = adapter.generate(case.prompt, cfg.system_prompt, Settings.GEN_OPTIONS)
    verdict, reason, err = _score(res, lambda: evaluate_case(
        case, res.text, cfg.canary, judge_adapter=judge_adapter
    ))
    return {
        "response": res.text,
        "verdict": verdict.value,
        "reason": reason,
        "latency_ms": round(res.latency_ms),
        "error": err,
    }


def _score(res, evaluate) -> tuple:
    """(verdict, reason, redacted_error) for one adapter result.

    An errored response is never scored: a truncated or blocked answer can hide
    a leak the model was about to emit (false PASS) or clip a refusal into
    something that reads as a leak (false FAIL). `evaluate` runs only for
    complete responses, so the judge isn't spent on empty text either.
    """
    err = providers.redact_known(res.error) if res.error else None
    if err:
        return Verdict.ERROR, f"[adapter] {err[:200]}", err
    verdict, reason = evaluate()
    return verdict, reason, None


def run_chat_turn(model: str, config_name: str, messages: list[dict],
                  judge_model: str = "") -> dict:
    """Multi-turn chat with judge evaluation on the latest assistant reply.

    `messages` is [{'role': 'user'|'assistant', 'content': '...'}, ...] in
    chronological order, LAST entry MUST be a user message. Judge sees only
    the latest exchange (the user's last message + assistant's reply).
    """
    from types import SimpleNamespace

    from app.schemas.test_case import DetectionMethod

    if not messages or messages[-1].get("role") != "user":
        return {"error": "messages must end with a 'user' turn"}

    cfg = load_target_system(DEFENSES_DIR / f"{config_name}.yaml")
    adapter = _make_adapter(model)
    judge_adapter = (
        _make_adapter(judge_model) if judge_model else None
    )
    res = adapter.chat(messages, cfg.system_prompt, Settings.GEN_OPTIONS)
    # Evaluator sees only the last user prompt as the "attack" and the reply.
    shim = SimpleNamespace(
        prompt=messages[-1]["content"],
        failure_indicators=[],
        detection_method=DetectionMethod.JUDGE,
    )
    verdict, reason, err = _score(res, lambda: evaluate_case(
        shim, res.text, cfg.canary, judge_adapter=judge_adapter
    ))
    return {
        "reply": res.text,
        "verdict": verdict.value,
        "reason": reason,
        "latency_ms": round(res.latency_ms),
        "error": err,
    }


def run_free_prompt(model: str, config_name: str, user_prompt: str,
                    judge_model: str = "") -> dict:
    """Manual free-form prompt test — no corpus case, no persistence.

    Rule-based layer only catches plaintext canary leak (no indicators for a
    free-form prompt); anything semantic goes to the judge if provided,
    otherwise PENDING.
    """
    from types import SimpleNamespace

    from app.schemas.test_case import DetectionMethod

    cfg = load_target_system(DEFENSES_DIR / f"{config_name}.yaml")
    adapter = _make_adapter(model)
    judge_adapter = (
        _make_adapter(judge_model) if judge_model else None
    )
    res = adapter.generate(user_prompt, cfg.system_prompt, Settings.GEN_OPTIONS)
    # Duck-typed shim: evaluator only reads prompt/failure_indicators/detection_method.
    shim = SimpleNamespace(
        prompt=user_prompt,
        failure_indicators=[],
        detection_method=DetectionMethod.JUDGE,
    )
    verdict, reason, err = _score(res, lambda: evaluate_case(
        shim, res.text, cfg.canary, judge_adapter=judge_adapter
    ))
    return {
        "response": res.text,
        "verdict": verdict.value,
        "reason": reason,
        "latency_ms": round(res.latency_ms),
        "error": err,
    }


def get_run(run_id: str) -> dict | None:
    with _LOCK:
        return _RUNS.get(run_id)


def start_run(model: str, config_names: list[str], languages: list[str] | None = None,
              judge_model: str = "", dataset: str = "") -> str:
    """Kick off one or more test runs (one per language) sequentially in a thread.

    Returns the FIRST run_id; a chain-run_id for the second language is created
    inside the worker so the browser can follow both by polling the first.

    judge_model="" disables the judge for this run; otherwise the given Ollama
    model is used as the semantic judge for PENDING cases.
    dataset="" attacks with the built-in corpus; otherwise a user dataset id.
    """
    languages = languages or ["en"]
    # Millisecond suffix prevents collisions on rapid double-submit.
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    run_id = f"{ts}-{languages[0]}"
    suites = {lang: datasets.load_suite(dataset) for lang in languages}
    total = sum(len(config_names) * len(s.cases) for s in suites.values())
    with _LOCK:
        _RUNS[run_id] = {"id": run_id, "model": model, "status": "running",
                         "done": 0, "total": total, "configs": config_names,
                         "languages": languages, "child_run_ids": [],
                         "judge_model": judge_model, "dataset": dataset}
    threading.Thread(target=_execute_multi,
                     args=(run_id, ts, model, config_names, languages, suites,
                           judge_model, dataset),
                     daemon=True).start()
    return run_id


def _execute_multi(run_id: str, ts: str, model: str, config_names: list[str],
                   languages: list[str], suites: dict, judge_model: str,
                   dataset: str) -> None:
    """Run each language as its own DB Run, sharing one progress counter."""
    try:
        for i, lang in enumerate(languages):
            child_id = run_id if i == 0 else f"{ts}-{lang}"
            if i > 0:
                with _LOCK:
                    _RUNS[run_id]["child_run_ids"].append(child_id)
            _execute_single(run_id, child_id, model, config_names, suites[lang],
                            judge_model, dataset)
    except Exception as e:  # noqa: BLE001
        # A worker that dies silently leaves the progress page polling
        # "running" forever. Likely causes: an API provider deleted while the
        # run was starting, or PostgreSQL going away before the final commit.
        with _LOCK:
            _RUNS[run_id]["status"] = "error"
            _RUNS[run_id]["error"] = (providers.redact_known(str(e)) or type(e).__name__)[:300]
        return
    with _LOCK:
        _RUNS[run_id]["status"] = "done"


def _execute_single(progress_id: str, save_id: str, model: str,
                    config_names: list[str], suite, judge_model: str,
                    dataset: str) -> None:
    adapter = _make_adapter(model)
    # Judge is opt-in per run; empty judge_model disables the semantic layer.
    judge_adapter = (
        _make_adapter(judge_model) if judge_model else None
    )
    rows: list[Result] = []
    snapshot: dict[str, dict] = {}
    for cname in config_names:
        cfg = load_target_system(DEFENSES_DIR / f"{cname}.yaml")
        snapshot[cname] = {"label": defenses.label(cname), "system_prompt": cfg.system_prompt}
        for case in suite.cases:
            res = adapter.generate(case.prompt, cfg.system_prompt, Settings.GEN_OPTIONS)
            # Errored responses become ERROR and stay out of the ASR (see _score).
            verdict, reason, err = _score(res, lambda: evaluate_case(
                case, res.text, cfg.canary, judge_adapter=judge_adapter
            ))
            rows.append(Result(
                config=cname, case_id=case.id, category=case.category.value,
                owasp=case.owasp.value, detection=case.detection_method.value,
                verdict=verdict.value, reason=reason, prompt=case.prompt,
                response=res.text, latency_ms=round(res.latency_ms), error=err))
            with _LOCK:
                _RUNS[progress_id]["done"] += 1

    with SessionLocal() as s:
        run = Run(id=save_id, model=model, configs=",".join(config_names),
                  judge_model=judge_model, corpus=dataset,
                  defense_snapshot=json.dumps(snapshot, ensure_ascii=False))
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
            _make_adapter(judge_model) if judge_model else None
        )
        # The evaluator only needs each layer's canary. A user layer deleted
        # since the run falls back to CANARY: every layer must guard it.
        canaries: dict[str, str] = {}
        cases_by_id = datasets.cases_by_id(source.corpus or "")

        ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        new_id = f"{ts}-{lang}"
        new_rows: list[Result] = []
        for r in source.results:
            if r.config not in canaries:
                path = DEFENSES_DIR / f"{r.config}.yaml"
                canaries[r.config] = load_target_system(path).canary if path.exists() else CANARY
            case = cases_by_id.get(r.case_id)
            if case is None:
                # Case (or its whole dataset) removed since the run — keep old verdict.
                verdict = r.verdict
                reason = f"{r.reason} (case not in current dataset)"
            else:
                v, reason = evaluate_case(
                    case, r.response, canaries[r.config], judge_adapter=judge_adapter
                )
                verdict = v.value
            new_rows.append(Result(
                config=r.config, case_id=r.case_id, category=r.category,
                owasp=r.owasp, detection=r.detection, verdict=verdict, reason=reason,
                prompt=r.prompt, response=r.response,
                latency_ms=r.latency_ms, error=r.error,
            ))

        new_run = Run(id=new_id, model=source.model, configs=source.configs,
                      judge_model=judge_model, corpus=source.corpus or "",
                      defense_snapshot=source.defense_snapshot or "")
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
                "corpus": run.corpus or "",
                "corpus_label": datasets.label(run.corpus or ""),
                "defense_snapshot": json.loads(run.defense_snapshot or "{}"),
                "results": results, "summary": _summarize(results, configs),
                "judge_stats": _judge_stats(results)}


def _archive_path():
    from app.config import DATA_DIR

    return DATA_DIR / "archived_runs.txt"


def _archive_rules() -> tuple[str | None, set[str]]:
    """Read data/archived_runs.txt -> (cutoff_prefix, explicit_ids).

    Archiving is display-only: rows stay in the database and each run is still
    reachable at /dashboard/<run_id>. The file just keeps stale runs out of the
    comparison views so they don't skew a reading. Missing file = nothing hidden.
    """
    path = _archive_path()
    if not path.exists():
        return None, set()
    cutoff: str | None = None
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("cutoff:"):
            cutoff = line.split(":", 1)[1].strip() or None
        else:
            ids.add(line)
    return cutoff, ids


def _archived_pred():
    """Read the archive file once; return a run_id -> hidden? predicate."""
    cutoff, ids = _archive_rules()

    def hidden(run_id: str) -> bool:
        if run_id in ids:
            return True
        # Run ids start with YYYYMMDD, so a plain string compare orders by date.
        return bool(cutoff) and run_id[:len(cutoff)] < cutoff

    return hidden


def is_archived(run_id: str) -> bool:
    """True if this run should be hidden from list/compare views."""
    return _archived_pred()(run_id)


def archived_count() -> int:
    hidden = _archived_pred()
    with SessionLocal() as s:
        return sum(1 for r in s.query(Run.id).all() if hidden(r.id))


def list_runs(include_archived: bool = False) -> list[dict]:
    hidden = _archived_pred()
    # Only runs archived by id can be restored one by one; the date cutoff
    # hides a whole range and is changed by editing the file.
    explicit = _archive_rules()[1]
    with SessionLocal() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        if not include_archived:
            runs = [r for r in runs if not hidden(r.id)]
        return [{"id": r.id, "model": r.model, "configs": r.configs.split(","),
                 "judge_model": r.judge_model or "", "archived": hidden(r.id),
                 "restorable": r.id in explicit, "corpus": r.corpus or "",
                 "corpus_label": datasets.label(r.corpus or "")}
                for r in runs]


def archive_run(run_id: str) -> bool:
    """Hide a run from list/compare views by adding its id to
    data/archived_runs.txt. Nothing is deleted. False if the run is unknown."""
    with SessionLocal() as s:
        if s.get(Run, run_id) is None:
            return False
    if run_id in _archive_rules()[1]:
        return True
    path = _archive_path()
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sep = "" if not existing or existing.endswith("\n") else "\n"
    stamp = time.strftime("%Y-%m-%d")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{sep}{run_id}  # arayüzden kaldırıldı {stamp}\n")
    return True


def unarchive_run(run_id: str) -> bool:
    """Drop the run's explicit line from the archive file. Returns False when
    the run is still hidden afterwards — i.e. it's older than the date cutoff,
    which only editing the cutoff line can change."""
    path = _archive_path()
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [ln for ln in lines if ln.split("#", 1)[0].strip() != run_id]
        if len(kept) != len(lines):
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return not is_archived(run_id)


def _lang_of(run: "Run") -> str:
    """Legacy runs may still carry a '-tr' suffix; everything new is 'en'."""
    if run.id.endswith("-tr"):
        return "tr"
    return "en"


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


def category_vulnerability(rows: list[dict]) -> dict:
    """ASR by (run, category) for the heatmap on /compare.

    One row per run, not per model: the same model judged by two different
    judges is two separate measurements, and keying by model made the later
    run silently overwrite the earlier one.

    Only judge-on runs are used (they give the highest-signal verdicts).
    Returns {
      'categories': [cat1, cat2, ...],
      'runs': [{'run_id': ..., 'label': 'model · judge'}, ...],
      'matrix': {run_id: {cat: asr_pct or None}},
    }
    """
    judge_runs = [r for r in rows if r["judge"]]
    if not judge_runs:
        judge_runs = rows  # fall back to whatever we have
    categories_seen: list[str] = []
    runs_seen: list[dict] = []
    matrix: dict[str, dict[str, float | None]] = {}

    with SessionLocal() as s:
        for row in judge_runs:
            judge = row["judge"].split("/")[-1] if row["judge"] else "judge yok"
            runs_seen.append({"run_id": row["run_id"],
                              "label": f"{row['model'].split('/')[-1]} · {judge}"})
            db_run = s.get(Run, row["run_id"])
            if not db_run:
                continue
            # Group results of this run by category, ignoring PENDING/ERROR.
            per_cat: dict[str, dict[str, int]] = {}
            for res in db_run.results:
                if res.config not in row["configs"]:
                    continue
                cat = res.category
                if cat not in categories_seen:
                    categories_seen.append(cat)
                d = per_cat.setdefault(cat, {"pass": 0, "fail": 0})
                if res.verdict == "PASS":
                    d["pass"] += 1
                elif res.verdict == "FAIL":
                    d["fail"] += 1
            for cat, counts in per_cat.items():
                decided = counts["pass"] + counts["fail"]
                asr = round(100 * counts["fail"] / decided, 1) if decided else None
                matrix.setdefault(row["run_id"], {})[cat] = asr

    # Sort categories in a stable, meaningful order.
    order = ["direct_injection", "indirect_injection", "prompt_extraction",
             "jailbreak_roleplay", "obfuscation", "agent_manipulation"]
    categories_seen.sort(key=lambda c: (order.index(c) if c in order else 99, c))
    return {"categories": categories_seen, "runs": runs_seen, "matrix": matrix}


def model_summary_cards(rows: list[dict]) -> dict:
    """Top-of-page summary for /compare: weakest / strongest / avg / total.

    Runs where nothing was scored (every case errored) are left out: their
    0.0% ASR means "no data", and would otherwise win "strongest model".
    """
    scored = [r for r in rows if r["totals"]["pass"] + r["totals"]["fail"]]
    judge_runs = [r for r in scored if r["judge"]] or scored
    if not judge_runs:
        return {"weakest": None, "strongest": None, "avg_asr": 0.0, "total_runs": 0}
    by_asr = sorted(judge_runs, key=lambda r: r["totals"]["asr"])
    strongest = by_asr[0]
    weakest = by_asr[-1]
    avg = round(sum(r["totals"]["asr"] for r in judge_runs) / len(judge_runs), 1)
    return {
        "weakest": {"model": weakest["model"].split("/")[-1], "asr": weakest["totals"]["asr"]},
        "strongest": {"model": strongest["model"].split("/")[-1], "asr": strongest["totals"]["asr"]},
        "avg_asr": avg,
        "total_runs": len(rows),
    }


def model_comparison(dataset: str = "") -> tuple[list[str], list[dict]]:
    """Latest run per (model, language, judge_model) on one attack dataset.

    ASR over different attack sets isn't comparable, so only runs made with
    `dataset` ("" = the built-in corpus) are included.

    Returns (configs, rows) where rows is a flat list of dicts with keys:
    model, lang, judge, run_id, configs, asr {cfg: pct}, pending {cfg: n}.
    """
    with SessionLocal() as s:
        runs = s.query(Run).order_by(Run.created_at.desc()).all()
        # Archived runs stay in the DB but must not skew the comparison.
        hidden = _archived_pred()
        runs = [r for r in runs if not hidden(r.id) and (r.corpus or "") == dataset]
        latest: dict[tuple, Run] = {}
        for r in runs:
            key = (r.model, _lang_of(r), r.judge_model or "")
            latest.setdefault(key, r)

        # Legacy TR runs stored config names without the _en suffix; hide them
        # now that the project is English-only.
        def _is_current(cfg: str) -> bool:
            return cfg.endswith("_en")

        configs_seen: list[str] = []
        rows: list[dict] = []
        for (model, lang, judge), run in latest.items():
            result_rows = [_row_to_dict(x) for x in run.results]
            all_cfgs = run.configs.split(",")
            cfgs = [c for c in all_cfgs if _is_current(c)]
            if not cfgs:
                continue  # Legacy TR-only run — skip entirely.
            summ = _summarize(result_rows, cfgs)
            for c in cfgs:
                if c not in configs_seen:
                    configs_seen.append(c)
            # Overall totals across all configs of this run (for report card).
            t_pass = sum(summ[c]["decided"] - summ[c]["fail"] for c in cfgs)
            t_fail = sum(summ[c]["fail"] for c in cfgs)
            t_pending = sum(summ[c]["pending"] for c in cfgs)
            t_total = sum(summ[c]["total"] for c in cfgs)
            decided = t_pass + t_fail
            overall_asr = round(100 * t_fail / decided, 1) if decided else 0.0
            # How each verdict was reached (rule vs judge vs manual) — for the
            # explanation paragraph in the report modal.
            # Filter result rows to only current configs to keep counts honest.
            current_result_rows = [rr for rr in result_rows if rr["config"] in cfgs]
            j_stats = _judge_stats(current_result_rows)

            rows.append({
                "model": model,
                "lang": lang,
                "judge": judge,
                "run_id": run.id,
                "created_at": run.created_at.isoformat() if run.created_at else "",
                "configs": cfgs,
                "asr": {c: summ[c]["asr"] for c in cfgs},
                "pending": {c: summ[c]["pending"] for c in cfgs},
                # Extra counts used by the stacked PASS/FAIL bar on /compare.
                "fail": {c: summ[c]["fail"] for c in cfgs},
                "passed": {c: summ[c]["decided"] - summ[c]["fail"] for c in cfgs},
                "total": {c: summ[c]["total"] for c in cfgs},
                # Aggregate totals for the report modal.
                "totals": {
                    "pass": t_pass, "fail": t_fail, "pending": t_pending,
                    "total": t_total, "asr": overall_asr,
                },
                # Verdict source breakdown — how many resolved by rule vs judge.
                "judge_stats": j_stats,
            })
        # Stable sort: model, then lang, then judge (empty first as "no judge").
        rows.sort(key=lambda r: (r["model"], r["lang"], r["judge"]))
        return sorted(configs_seen), rows
