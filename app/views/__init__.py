"""L6 · Presentation — Flask routes.

Screens: Run (pick model + defense configs, trigger), Progress (live poll),
Dashboard (ASR chart + results table), Detail (one attack/response).
"""

from __future__ import annotations

from flask import (Blueprint, jsonify, redirect, render_template, request,
                   url_for)

from app import runner
from app.adapters.ollama import OllamaAdapter
from app.config import Settings

bp = Blueprint("main", __name__)


# Human-readable descriptions shown next to each defense config checkbox.
# Add a new entry when a new config{N}_{name}_en.yaml file is introduced.
_CONFIG_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "config0_none_en":     ("None (baseline)", "Sadece canary tanımlı, kural yok"),
    "config1_basic_en":    ("Basic",           "4 kısa kural (SafeAssist persona)"),
    "config2_hardened_en": ("Hardened",        "4 sıkı kural (instruction hierarchy, encoding resistance, confidentiality, pretext immunity)"),
}


def _target_models() -> list[str]:
    """List target-eligible models: installed Ollama (minus judges) + Gemini."""
    all_models = OllamaAdapter.list_models(Settings.OLLAMA_HOST)
    judges = set(Settings.JUDGE_MODELS)
    ollama_targets = [
        m for m in all_models
        if m not in judges and not any(m.startswith(f"{j}:") for j in judges)
    ]
    # Add Gemini models if the API key is configured — tagged with 'gemini:'
    # so the runner's adapter factory can dispatch correctly.
    gemini_targets: list[str] = []
    if Settings.GEMINI_API_KEY:
        try:
            from app.adapters.gemini import KNOWN_GEMINI_MODELS
            gemini_targets = [f"gemini:{m}" for m in KNOWN_GEMINI_MODELS]
        except ImportError:
            pass
    return ollama_targets + gemini_targets


def _judge_model_options() -> list[dict]:
    """List installed Ollama models as judge candidates, plus a 'none' option.

    The default suggestion (Settings.JUDGE_MODEL) is surfaced first if present.
    """
    all_models = OllamaAdapter.list_models(Settings.OLLAMA_HOST)
    default = Settings.JUDGE_MODEL
    default_variants = [m for m in all_models
                        if m == default or m.startswith(f"{default}:")]
    others = [m for m in all_models if m not in default_variants]
    ordered = default_variants + others
    return [{"value": m, "label": m} for m in ordered]


def _defense_configs_with_desc() -> list[dict]:
    """Return [{name, label, desc}] for all defense YAMLs found on disk."""
    out = []
    for name in runner.list_defense_configs():
        label, desc = _CONFIG_DESCRIPTIONS.get(name, (name, ""))
        out.append({"name": name, "label": label, "desc": desc})
    return out


@bp.route("/")
def index():
    models = _target_models()
    return render_template(
        "index.html",
        models=models,
        configs=_defense_configs_with_desc(),
        judge_options=_judge_model_options(),
        default_judge=Settings.JUDGE_MODEL,
        runs=runner.list_runs(),
        ollama_up=bool(models),
    )


@bp.route("/run", methods=["POST"])
def run():
    model = request.form.get("model", "")
    configs = request.form.getlist("configs")
    # Judge model: empty string means judge is off for this run.
    judge_model = request.form.get("judge_model", "").strip()
    if not model or not configs:
        return redirect(url_for("main.index"))
    # English-only corpus; language dimension removed 2026-08-24.
    run_id = runner.start_run(model, configs, ["en"], judge_model=judge_model)
    return redirect(url_for("main.progress", run_id=run_id))


@bp.route("/progress/<run_id>")
def progress(run_id):
    return render_template("progress.html", run_id=run_id)


@bp.route("/progress/<run_id>/status")
def progress_status(run_id):
    st = runner.get_run(run_id)
    if not st:
        return jsonify({"status": "unknown"})
    return jsonify({"status": st["status"], "done": st["done"], "total": st["total"]})


@bp.route("/compare")
def compare():
    configs, rows = runner.model_comparison()
    # Belt & suspenders: never show legacy TR-only config columns even if a
    # stale DB row still carries them.
    configs = [c for c in configs if c.endswith("_en")]
    pairs = runner.judge_impact_pairs(rows)
    return render_template("compare.html", configs=configs, rows=rows, pairs=pairs)


@bp.route("/dashboard/<run_id>")
def dashboard(run_id):
    run = runner.load_run(run_id)
    if not run:
        return redirect(url_for("main.index"))
    # Aggregate totals across all configs for the top stat cards.
    totals = {"cases": 0, "pass": 0, "fail": 0, "pending": 0}
    for s in run["summary"].values():
        totals["cases"]   += s["total"]
        totals["pass"]    += s["decided"] - s["fail"]
        totals["fail"]    += s["fail"]
        totals["pending"] += s["pending"]
    decided = totals["pass"] + totals["fail"]
    totals["asr"] = round(100 * totals["fail"] / decided, 1) if decided else 0.0
    return render_template(
        "dashboard.html",
        run=run,
        totals=totals,
        judge_options=_judge_model_options(),
        default_judge=Settings.JUDGE_MODEL,
    )


@bp.route("/run/<run_id>/re-evaluate", methods=["POST"])
def re_evaluate(run_id):
    """Re-score an existing run with a different judge (or none)."""
    judge_model = request.form.get("judge_model", "").strip()
    new_run_id = runner.re_evaluate_run(run_id, judge_model)
    if not new_run_id:
        return redirect(url_for("main.dashboard", run_id=run_id))
    return redirect(url_for("main.dashboard", run_id=new_run_id))


@bp.route("/case/<run_id>/<int:idx>")
def case_detail(run_id, idx):
    run = runner.load_run(run_id)
    if not run or idx >= len(run["results"]):
        return redirect(url_for("main.index"))
    return render_template("detail.html", run=run, row=run["results"][idx], idx=idx)


@bp.route("/result/<int:pk>/finalize", methods=["POST"])
def finalize_result(pk):
    """Manually override a PENDING verdict (PASS/FAIL/ERROR)."""
    verdict = request.form.get("verdict", "")
    result = runner.finalize_result(pk, verdict)
    if result is None:
        return redirect(url_for("main.index"))
    # Redirect back to the same detail page so the user sees the update.
    run = runner.load_run(result["run_id"])
    if run:
        for idx, row in enumerate(run["results"]):
            if row["pk"] == pk:
                return redirect(url_for("main.case_detail", run_id=result["run_id"], idx=idx))
    return redirect(url_for("main.dashboard", run_id=result["run_id"]))


@bp.route("/corpus")
def corpus_list():
    cases_by_lang = runner.list_all_cases()
    grouped_en: dict[str, list] = {}
    for c in cases_by_lang.get("en", []):
        grouped_en.setdefault(c.category.value, []).append(c)
    return render_template(
        "corpus_list.html",
        grouped_en=grouped_en,
        total_en=sum(len(v) for v in grouped_en.values()),
    )


@bp.route("/corpus/<lang>/<case_id>", methods=["GET"])
def corpus_case(lang, case_id):
    case = runner.get_case(lang, case_id)
    if case is None:
        return redirect(url_for("main.corpus_list"))
    return render_template(
        "corpus_case.html",
        case=case,
        lang=lang,
        models=_target_models(),
        configs=runner.list_defense_configs(),
        judge_options=_judge_model_options(),
        default_judge=Settings.JUDGE_MODEL,
    )


@bp.route("/corpus/<lang>/<case_id>/test", methods=["POST"])
def corpus_case_test(lang, case_id):
    model = request.form.get("model", "")
    config_name = request.form.get("config", "")
    judge_model = request.form.get("judge_model", "").strip()
    if not model or not config_name:
        return jsonify({"error": "model and config are required"}), 400
    result = runner.run_single_attack(lang, case_id, model, config_name, judge_model)
    return jsonify(result)


@bp.route("/architecture")
def architecture():
    return render_template("architecture.html")


@bp.route("/manual", methods=["GET"])
def manual_test():
    return render_template(
        "manual.html",
        models=_target_models(),
        configs=runner.list_defense_configs(),
        judge_options=_judge_model_options(),
        default_judge=Settings.JUDGE_MODEL,
    )


@bp.route("/manual/test", methods=["POST"])
def manual_test_run():
    model = request.form.get("model", "")
    config_name = request.form.get("config", "")
    judge_model = request.form.get("judge_model", "").strip()
    user_prompt = request.form.get("prompt", "").strip()
    if not model or not config_name or not user_prompt:
        return jsonify({"error": "model, config, and prompt are required"}), 400
    result = runner.run_free_prompt(model, config_name, user_prompt, judge_model)
    return jsonify(result)


@bp.route("/manual/chat", methods=["POST"])
def manual_chat():
    """Multi-turn chat turn. Body: {model, config, judge_model, messages: [...]}"""
    data = request.get_json(silent=True) or {}
    model = (data.get("model") or "").strip()
    config_name = (data.get("config") or "").strip()
    judge_model = (data.get("judge_model") or "").strip()
    messages = data.get("messages") or []
    if not model or not config_name:
        return jsonify({"error": "model and config are required"}), 400
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400
    # Basic shape check.
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            return jsonify({"error": "each message needs role user|assistant"}), 400
    result = runner.run_chat_turn(model, config_name, messages, judge_model)
    return jsonify(result)
