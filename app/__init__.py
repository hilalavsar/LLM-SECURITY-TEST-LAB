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


def create_app() -> Flask:
    app = Flask(__name__)

    from app.db import init_db
    try:
        init_db()
    except Exception as e:  # noqa: BLE001 - app should still boot if DB is down
        app.logger.warning("DB init failed (is PostgreSQL up?): %s", e)

    from app.views import bp
    app.register_blueprint(bp)
    return app
