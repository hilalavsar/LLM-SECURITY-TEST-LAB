"""Export comparison + case-level results as static JSON for the HF Space.

Reuses app.runner's existing aggregation (model_comparison, load_run, ...)
against the project's own Postgres DB, and writes read-only snapshots into
../hf_space/data/ — a sibling folder OUTSIDE this git repo (kept separate on
purpose: this repo goes to GitHub, hf_space/ goes to its own HF Space repo,
and the two should never share a push). Run this locally after each test
round, then push the hf_space/ folder to the HF Space repo — the Space
itself never touches the DB or Ollama, it only reads these two JSON files.

Usage:
    python scripts/export_hf_space.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app import runner  # noqa: E402

OUT_DIR = REPO_ROOT.parent / "hf_space" / "data"

# FAIL = the attack succeeded. Don't republish the model's full harmful
# completion on a public Space — keep enough to show the verdict was
# reached correctly, redact the rest. PASS/REVIEW responses are shown in full.
FAIL_RESPONSE_PREVIEW_CHARS = 200
REDACTION_NOTE = " […kısaltıldı, başarılı saldırı çıktısı tam yayınlanmıyor]"


def _redact_response(row: dict) -> dict:
    if row["verdict"] != "FAIL":
        return row
    text = row.get("response") or ""
    if len(text) <= FAIL_RESPONSE_PREVIEW_CHARS:
        return row
    redacted = dict(row)
    redacted["response"] = text[:FAIL_RESPONSE_PREVIEW_CHARS] + REDACTION_NOTE
    return redacted


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs, comparison_rows = runner.model_comparison()
    summary_cards = runner.model_summary_cards(comparison_rows)
    vulnerability = runner.category_vulnerability(comparison_rows)

    cases: list[dict] = []
    for row in comparison_rows:
        run = runner.load_run(row["run_id"])
        if not run:
            continue
        for result in run["results"]:
            if result["config"] not in row["configs"]:
                continue  # legacy/non-current config, excluded from comparison
            cases.append({
                **_redact_response(result),
                "model": row["model"],
                "lang": row["lang"],
                "judge_model": run["judge_model"],
                "run_id": row["run_id"],
            })

    (OUT_DIR / "summary.json").write_text(
        json.dumps({
            "configs": configs,
            "comparison": comparison_rows,
            "summary_cards": summary_cards,
            "vulnerability": vulnerability,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{len(comparison_rows)} model satırı, {len(cases)} vaka kaydı yazıldı -> {OUT_DIR}")


if __name__ == "__main__":
    main()
