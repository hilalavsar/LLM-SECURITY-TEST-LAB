"""LLM Security Test Lab — Flask application package.

Katman haritasi:
  L5/L6  app/  (bu paket)      -> Flask app + views + templates
  L4     app/runner/           -> test runner (CLI)
  L3     app/evaluator/        -> rules + judge
  L2     app/adapters/         -> BaseModelAdapter + Ollama/Mock
  L1     (harici) Ollama daemon
  Persistence  app/models/ + app/schemas/
"""
