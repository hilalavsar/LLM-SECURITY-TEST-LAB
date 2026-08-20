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
# Add a new entry when a new config{N}_{name}.yaml file is introduced.
_CONFIG_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # Turkish defense prompts
    "config0_none":        ("Yok (baseline) · TR",  "Sadece canary tanımlı, kural yok"),
    "config1_basic":       ("Temel · TR",           "4 kısa kural (paylaşma, açıklama, zararlı reddet, rolü değiştirme)"),
    "config2_hardened":    ("Sıkı · TR",            "4 sıkı kural (instruction priority, format coverage, secrecy, pretext immunity)"),
    # English defense prompts (2x2 experiment)
    "config0_none_en":     ("None (baseline) · EN", "Only canary defined, no rules — English"),
    "config1_basic_en":    ("Basic · EN",           "4 short rules — English (SafeAssist persona)"),
    "config2_hardened_en": ("Hardened · EN",        "4 strict rules — English (instruction hierarchy, encoding resistance, ...)"),
}


def _target_models() -> list[str]:
    """List installed Ollama models minus the judge (judge is not a test target)."""
    all_models = OllamaAdapter.list_models(Settings.OLLAMA_HOST)
    judge_name = Settings.JUDGE_MODEL
    return [m for m in all_models
            if m != judge_name and not m.startswith(f"{judge_name}:")]


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
    """Return [{name, label, desc, lang}] so templates can group by language."""
    out = []
    for name in runner.list_defense_configs():
        label, desc = _CONFIG_DESCRIPTIONS.get(name, (name, ""))
        lang = "en" if name.endswith("_en") else "tr"
        out.append({"name": name, "label": label, "desc": desc, "lang": lang})
    return out


def _defense_configs_grouped() -> dict:
    """Group configs by language for two-column UI: {'tr': [...], 'en': [...]}"""
    grouped: dict = {"tr": [], "en": []}
    for c in _defense_configs_with_desc():
        grouped[c["lang"]].append(c)
    return grouped


@bp.route("/")
def index():
    models = _target_models()
    return render_template(
        "index.html",
        models=models,
        configs_by_lang=_defense_configs_grouped(),
        judge_options=_judge_model_options(),
        default_judge=Settings.JUDGE_MODEL,
        runs=runner.list_runs(),
        ollama_up=bool(models),
    )


@bp.route("/run", methods=["POST"])
def run():
    model = request.form.get("model", "")
    configs = request.form.getlist("configs")
    # Language is now user-choice: TR, EN, or both — at least one required.
    languages = [l for l in request.form.getlist("lang") if l in ("tr", "en")]
    # Judge model: empty string means judge is off for this run.
    judge_model = request.form.get("judge_model", "").strip()
    if not model or not configs or not languages:
        return redirect(url_for("main.index"))
    run_id = runner.start_run(model, configs, languages, judge_model=judge_model)
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
    pairs = runner.judge_impact_pairs(rows)
    return render_template("compare.html", configs=configs, rows=rows, pairs=pairs)


@bp.route("/dashboard/<run_id>")
def dashboard(run_id):
    run = runner.load_run(run_id)
    if not run:
        return redirect(url_for("main.index"))
    return render_template(
        "dashboard.html",
        run=run,
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
    # Group each language's cases by category for readable rendering.
    def group_by_cat(cases):
        grouped: dict[str, list] = {}
        for c in cases:
            grouped.setdefault(c.category.value, []).append(c)
        return grouped
    grouped_tr = group_by_cat(cases_by_lang.get("tr", []))
    grouped_en = group_by_cat(cases_by_lang.get("en", []))
    return render_template(
        "corpus_list.html",
        grouped_tr=grouped_tr,
        grouped_en=grouped_en,
        total_tr=sum(len(v) for v in grouped_tr.values()),
        total_en=sum(len(v) for v in grouped_en.values()),
    )


@bp.route("/corpus/<lang>/<case_id>", methods=["GET"])
def corpus_case(lang, case_id):
    case = runner.get_case(lang, case_id)
    if case is None:
        return redirect(url_for("main.corpus_list"))
    models = _target_models()
    configs = runner.list_defense_configs()
    return render_template(
        "corpus_case.html",
        case=case,
        lang=lang,
        models=models,
        configs=configs,
    )


@bp.route("/corpus/<lang>/<case_id>/test", methods=["POST"])
def corpus_case_test(lang, case_id):
    model = request.form.get("model", "")
    config_name = request.form.get("config", "")
    if not model or not config_name:
        return jsonify({"error": "model and config are required"}), 400
    result = runner.run_single_attack(lang, case_id, model, config_name)
    return jsonify(result)
