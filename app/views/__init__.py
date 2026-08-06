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


@bp.route("/")
def index():
    models = OllamaAdapter.list_models(Settings.OLLAMA_HOST)
    return render_template(
        "index.html",
        models=models,
        configs=runner.list_defense_configs(),
        runs=runner.list_runs(),
        ollama_up=bool(models),
    )


@bp.route("/run", methods=["POST"])
def run():
    model = request.form.get("model", "")
    configs = request.form.getlist("configs")
    if not model or not configs:
        return redirect(url_for("main.index"))
    # TR always runs; EN is opt-in via checkbox.
    languages = ["tr"]
    if request.form.get("also_en"):
        languages.append("en")
    run_id = runner.start_run(model, configs, languages)
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
    configs, languages, data = runner.model_comparison()
    return render_template("compare.html", configs=configs,
                           languages=languages, data=data)


@bp.route("/dashboard/<run_id>")
def dashboard(run_id):
    run = runner.load_run(run_id)
    if not run:
        return redirect(url_for("main.index"))
    return render_template("dashboard.html", run=run)


@bp.route("/case/<run_id>/<int:idx>")
def case_detail(run_id, idx):
    run = runner.load_run(run_id)
    if not run or idx >= len(run["results"]):
        return redirect(url_for("main.index"))
    return render_template("detail.html", run=run, row=run["results"][idx], idx=idx)
