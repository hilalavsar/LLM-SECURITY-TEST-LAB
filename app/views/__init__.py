"""L6 · Presentation — Flask routes.

Screens: Run (pick model + defense configs, trigger), Progress (live poll),
Dashboard (ASR chart + results table), Detail (one attack/response),
Providers (OpenAI-compatible endpoints added at runtime, keys in memory only).
"""

from __future__ import annotations

from flask import (Blueprint, Response, abort, jsonify, redirect,
                   render_template, request, url_for)

from app import datasets, defenses, providers, reporter, runner
from app.adapters.ollama import OllamaAdapter
from app.config import Settings
from app.schemas.test_case import CANARY

bp = Blueprint("main", __name__)


def _target_models() -> list[str]:
    """List target-eligible models: installed Ollama (minus judges), Gemini,
    and models of API providers added from the UI in this session."""
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
            from app.adapters.gemini import GeminiAdapter
            # Discover models actually supported by this API key; cached.
            gemini_targets = [f"gemini:{m}" for m in GeminiAdapter.list_models()]
        except ImportError:
            pass
    # Runtime providers are tagged 'api:<provider>:<model>'.
    api_targets = [c["value"] for c in providers.model_choices()]
    return ollama_targets + gemini_targets + api_targets


def _summary_model_options() -> list[dict]:
    """Models that can write the AI summary. Judges are left out: the
    fine-tuned judge is trained to emit one-line verdicts, not reports."""
    api_labels = {c["value"]: c["label"] for c in providers.model_choices()}
    return [{"value": m, "label": api_labels.get(m, m)} for m in _target_models()]


def _judge_model_options() -> list[dict]:
    """Judge candidates: Ollama models, Gemini (if API key set), and runtime
    API providers.

    The default suggestion (Settings.JUDGE_MODEL) is surfaced first if present.
    """
    all_models = OllamaAdapter.list_models(Settings.OLLAMA_HOST)
    default = Settings.JUDGE_MODEL
    default_variants = [m for m in all_models
                        if m == default or m.startswith(f"{default}:")]
    others = [m for m in all_models if m not in default_variants]
    ordered = default_variants + others

    # Add Gemini judge candidates if API key configured
    if Settings.GEMINI_API_KEY:
        try:
            from app.adapters.gemini import GeminiAdapter
            for m in GeminiAdapter.list_models():
                ordered.append(f"gemini:{m}")
        except ImportError:
            pass

    options = [{"value": m, "label": m} for m in ordered]
    # API providers carry a readable label, e.g. "Groq · llama-3.3-70b-versatile".
    return options + providers.model_choices()


@bp.route("/")
def index():
    models = _target_models()
    # ?all=1 → arşivlenmiş koşuları da göster (data/archived_runs.txt).
    show_all = request.args.get("all") == "1"
    return render_template(
        "index.html",
        models=models,
        configs=defenses.list_layers(),
        datasets=datasets.list_datasets(),
        selected_dataset=request.args.get("dataset", ""),
        judge_options=_judge_model_options(),
        default_judge=Settings.JUDGE_MODEL,
        runs=runner.list_runs(include_archived=show_all),
        archived_count=runner.archived_count(),
        show_all=show_all,
        ollama_up=bool(models),
    )


@bp.route("/run", methods=["POST"])
def run():
    model = request.form.get("model", "")
    configs = request.form.getlist("configs")
    # Judge model: empty string means judge is off for this run.
    judge_model = request.form.get("judge_model", "").strip()
    dataset = request.form.get("dataset", "")
    known_configs = set(runner.list_defense_configs())
    if (not model or not configs or not datasets.exists(dataset)
            or not set(configs) <= known_configs):
        return redirect(url_for("main.index"))
    try:
        # Fail here, not inside the background thread, when an API provider
        # from an earlier session is gone (keys are memory-only).
        runner.check_models(model, judge_model)
    except providers.ProviderError as e:
        return redirect(url_for("main.providers_page", error=str(e)))
    # English-only corpus; language dimension removed 2026-08-24.
    run_id = runner.start_run(model, configs, ["en"], judge_model=judge_model,
                              dataset=dataset)
    return redirect(url_for("main.progress", run_id=run_id))


@bp.route("/progress/<run_id>")
def progress(run_id):
    return render_template("progress.html", run_id=run_id)


@bp.route("/progress/<run_id>/status")
def progress_status(run_id):
    st = runner.get_run(run_id)
    if not st:
        return jsonify({"status": "unknown"})
    return jsonify({"status": st["status"], "done": st["done"], "total": st["total"],
                    "error": st.get("error", "")})


@bp.route("/compare")
def compare():
    dataset = request.args.get("dataset", "")
    if not datasets.exists(dataset):
        dataset = ""
    configs, rows = runner.model_comparison(dataset)
    configs = [c for c in configs if c.endswith("_en")]
    summary = runner.model_summary_cards(rows)
    vuln = runner.category_vulnerability(rows)
    return render_template(
        "compare.html",
        configs=configs, rows=rows,
        summary=summary, vuln=vuln,
        datasets=datasets.list_datasets(), dataset=dataset,
    )


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
    summary_options = _summary_model_options()
    # Cloud models write long reports faster and better than a 7B on 6 GB VRAM.
    default_summary = next((o["value"] for o in summary_options
                            if o["value"].startswith(("gemini:", providers.PREFIX))), "")
    return render_template(
        "dashboard.html",
        run=run,
        totals=totals,
        judge_options=_judge_model_options(),
        default_judge=Settings.JUDGE_MODEL,
        summary_options=summary_options,
        default_summary=default_summary,
        summary_langs=reporter.LANGS,
        summaries=reporter.for_run(run_id),
    )


@bp.route("/dashboard/<run_id>/all")
def dashboard_all(run_id):
    """Tek sayfada tüm case'ler (attack + response + verdict + reason)."""
    run = runner.load_run(run_id)
    if not run:
        return redirect(url_for("main.index"))
    # Totals for the header stat strip.
    totals = {"cases": 0, "pass": 0, "fail": 0, "pending": 0, "error": 0}
    for r in run["results"]:
        totals["cases"] += 1
        v = r.get("verdict", "")
        if v == "PASS": totals["pass"] += 1
        elif v == "FAIL": totals["fail"] += 1
        elif v == "PENDING": totals["pending"] += 1
        elif v == "ERROR": totals["error"] += 1
    decided = totals["pass"] + totals["fail"]
    totals["asr"] = round(100 * totals["fail"] / decided, 1) if decided else 0.0
    return render_template("dashboard_all.html", run=run, totals=totals)


@bp.route("/dashboard/<run_id>/ai-summary", methods=["POST"])
def ai_summary_create(run_id):
    run = runner.load_run(run_id)
    model = request.form.get("model", "").strip()
    lang = request.form.get("lang", "tr")
    if not run or not model or lang not in reporter.LANGS:
        return redirect(url_for("main.dashboard", run_id=run_id))
    try:
        reporter.generate(run, model, lang)
    except providers.ProviderError as e:
        return redirect(url_for("main.providers_page", error=str(e)))
    # Redirect so a page refresh shows the cached summary instead of re-running it.
    return redirect(url_for("main.ai_summary", run_id=run_id, model=model, lang=lang))


@bp.route("/dashboard/<run_id>/ai-summary", methods=["GET"])
def ai_summary(run_id):
    lang = request.args.get("lang", "tr")
    entry = reporter.get(run_id, request.args.get("model", ""), lang)
    run = runner.load_run(run_id)
    # Summaries are memory-only; after a restart old links fall back here.
    if not run or entry is None:
        return redirect(url_for("main.dashboard", run_id=run_id))
    return render_template(
        "ai_summary.html", run=run, entry=entry,
        body=reporter.to_html(entry["text"]), lang_name=reporter.LANGS[lang],
    )


@bp.route("/dashboard/<run_id>/ai-summary.md")
def ai_summary_download(run_id):
    lang = request.args.get("lang", "tr")
    entry = reporter.get(run_id, request.args.get("model", ""), lang)
    if entry is None or not entry["text"]:
        return redirect(url_for("main.dashboard", run_id=run_id))
    return Response(
        reporter.markdown_file(run_id, entry),
        mimetype="text/markdown; charset=utf-8",
        headers={"Content-Disposition":
                 f'attachment; filename="ai-ozet-{run_id}-{lang}.md"'},
    )


@bp.route("/run/<run_id>/archive", methods=["POST"])
def run_archive(run_id):
    """Hide a run from lists and comparisons (reversible, nothing deleted)."""
    runner.archive_run(run_id)
    # Fixed targets only — never redirect to a URL taken from the form.
    if request.form.get("back") == "index":
        return redirect(url_for("main.index"))
    return _back_to_compare()


@bp.route("/runs/archive", methods=["POST"])
def runs_archive():
    """Bulk version of run_archive, used by the compare page's run picker."""
    for run_id in request.form.getlist("run_id")[:500]:
        runner.archive_run(run_id)  # unknown ids are ignored
    return _back_to_compare()


def _back_to_compare():
    """Return to the compare tab the form came from (validated dataset id)."""
    dataset = request.form.get("dataset", "")
    if dataset and datasets.exists(dataset):
        return redirect(url_for("main.compare", dataset=dataset))
    return redirect(url_for("main.compare"))


@bp.route("/run/<run_id>/unarchive", methods=["POST"])
def run_unarchive(run_id):
    runner.unarchive_run(run_id)
    return redirect(url_for("main.index", all=1))


@bp.route("/run/<run_id>/re-evaluate", methods=["POST"])
def re_evaluate(run_id):
    """Re-score an existing run with a different judge (or none)."""
    judge_model = request.form.get("judge_model", "").strip()
    try:
        new_run_id = runner.re_evaluate_run(run_id, judge_model)
    except providers.ProviderError as e:
        return redirect(url_for("main.providers_page", error=str(e)))
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
    try:
        result = runner.run_single_attack(lang, case_id, model, config_name, judge_model)
    except providers.ProviderError as e:
        return jsonify({"error": str(e)}), 400
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
    try:
        result = runner.run_free_prompt(model, config_name, user_prompt, judge_model)
    except providers.ProviderError as e:
        return jsonify({"error": str(e)}), 400
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
    if not messages or not isinstance(messages, list):
        return jsonify({"error": "messages must be a non-empty list"}), 400
    # Basic shape check.
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            return jsonify({"error": "each message needs role user|assistant"}), 400
    try:
        result = runner.run_chat_turn(model, config_name, messages, judge_model)
    except providers.ProviderError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(result)


# --- API providers (runtime, memory-only) ------------------------------------


def _render_providers(error: str = "", form: dict | None = None, status: int = 200):
    added_id = request.args.get("added", "")
    return render_template(
        "providers.html",
        providers=providers.list_providers(),
        presets=providers.PRESETS,
        added_provider=providers.get_provider(added_id) if added_id else None,
        error=error or request.args.get("error", ""),
        ok=request.args.get("ok", ""),
        form=form or {},
        disabled=not Settings.ALLOW_UI_PROVIDERS,
    ), status


@bp.route("/providers", methods=["GET"])
def providers_page():
    return _render_providers()


@bp.route("/providers", methods=["POST"])
def providers_add():
    if not Settings.ALLOW_UI_PROVIDERS:
        abort(403)
    # The key is read once and never echoed back, not even on a form error.
    form = {
        "name": request.form.get("name", ""),
        "base_url": request.form.get("base_url", ""),
        "models": request.form.get("models", ""),
    }
    try:
        provider = providers.add_provider(
            form["name"], form["base_url"], request.form.get("api_key", ""),
            form["models"],
        )
    except providers.ProviderError as e:
        return _render_providers(error=str(e), form=form, status=400)
    return redirect(url_for("main.providers_page", added=provider.id))


@bp.route("/providers/<provider_id>/delete", methods=["POST"])
def providers_delete(provider_id):
    providers.remove_provider(provider_id)
    return redirect(url_for("main.providers_page"))


@bp.route("/providers/clear", methods=["POST"])
def providers_clear():
    providers.clear_providers()
    return redirect(url_for("main.providers_page"))


@bp.route("/providers/ollama-create", methods=["POST"])
def providers_ollama_create():
    """Register a local .gguf file into Ollama from a filesystem path."""
    name = request.form.get("name", "")
    gguf_path = request.form.get("gguf_path", "")
    try:
        OllamaAdapter.create_from_gguf(name, gguf_path)
    except ValueError as e:
        return redirect(url_for("main.providers_page", error=str(e)))
    return redirect(url_for(
        "main.providers_page",
        ok=f"'{name.strip()}' modeli eklendi — hedef ve judge listelerinde görünür.",
    ))


# --- attack datasets + defense layers (user-editable) ------------------------


def _writes_allowed() -> None:
    # Same switch as API providers: both let a visitor write to the server,
    # so one setting turns all of it off on a public deployment.
    if not Settings.ALLOW_UI_PROVIDERS:
        abort(403)


def _render_datasets(error: str = "", status: int = 200):
    return render_template(
        "datasets.html",
        datasets=datasets.list_datasets(),
        error=error,
        ok=request.args.get("ok", ""),
        max_rows=datasets.MAX_ROWS,
        disabled=not Settings.ALLOW_UI_PROVIDERS,
    ), status


@bp.route("/datasets", methods=["GET"])
def datasets_page():
    return _render_datasets()


@bp.route("/datasets", methods=["POST"])
def datasets_upload():
    _writes_allowed()
    f = request.files.get("file")
    if f is None or not f.filename:
        return _render_datasets(error="Bir dosya seç.", status=400)
    try:
        info = datasets.save_upload(request.form.get("name", ""), f.filename, f.read())
    except datasets.DatasetError as e:
        return _render_datasets(error=str(e), status=400)
    note = (f" {info['custom_category']} saldırının kategorisi belirtilmediği için "
            "'custom' atandı." if info["custom_category"] else "")
    return redirect(url_for(
        "main.datasets_page",
        ok=f"'{info['name']}' eklendi · {info['count']} saldırı.{note}",
    ))


@bp.route("/datasets/<dataset_id>/delete", methods=["POST"])
def datasets_delete(dataset_id):
    _writes_allowed()
    datasets.delete(dataset_id)
    return redirect(url_for("main.datasets_page"))


@bp.app_errorhandler(413)
def upload_too_large(_e):
    return _render_datasets(error="Dosya 5 MB sınırını aşıyor.", status=413)


def _render_defenses(error: str = "", form: dict | None = None, status: int = 200):
    return render_template(
        "defenses.html",
        layers=defenses.list_layers(),
        error=error,
        ok=request.args.get("ok", ""),
        form=form or {},
        canary=CANARY,
        disabled=not Settings.ALLOW_UI_PROVIDERS,
    ), status


@bp.route("/defenses", methods=["GET"])
def defenses_page():
    return _render_defenses()


@bp.route("/defenses", methods=["POST"])
def defenses_save():
    _writes_allowed()
    form = {k: request.form.get(k, "") for k in ("name", "label", "system_prompt")}
    try:
        name = defenses.save(form["label"], form["system_prompt"], form["name"])
    except defenses.DefenseError as e:
        return _render_defenses(error=str(e), form=form, status=400)
    return redirect(url_for("main.defenses_page",
                            ok=f"'{defenses.label(name)}' kaydedildi."))


@bp.route("/defenses/<name>/delete", methods=["POST"])
def defenses_delete(name):
    _writes_allowed()
    defenses.delete(name)
    return redirect(url_for("main.defenses_page"))
