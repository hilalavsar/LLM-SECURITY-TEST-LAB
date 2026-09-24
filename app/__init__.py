"""LLM Security Test Lab — Flask application package.

Layer map:
  L5/L6  app/  (this package)   -> Flask app + views + templates
  L4     app/runner/            -> test runner (background thread)
  L3     app/evaluator/         -> rules (+ judge later)
  L2     app/adapters/          -> BaseModelAdapter + Ollama/Mock
  L1     (external) Ollama daemon
  Persistence  app/schemas/ + data/runs/ (JSON for now, Postgres in Week 2)
"""

from __future__ import annotations

from flask import Flask

# Load .env before anything reads Settings.* — otherwise GEMINI_API_KEY,
# DATABASE_URL, etc. stay empty even though the file exists on disk.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can still come from the shell.


def create_app() -> Flask:
    app = Flask(__name__)
    # Dataset uploads are the only file uploads; a 500-row CSV is well under this.
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

    from app.db import init_db
    try:
        init_db()
    except Exception as e:  # noqa: BLE001 - app should still boot if DB is down
        app.logger.warning("DB init failed (is PostgreSQL up?): %s", e)

    # Jinja filter: `{{ 'config3_maximal_en' | savunma }}` → "Savunma 3 · Maksimum";
    # user layers show the name they were saved under.
    from app.defenses import label as defense_label

    app.add_template_filter(defense_label, "savunma")

    from app.views import bp
    app.register_blueprint(bp)
    return app
