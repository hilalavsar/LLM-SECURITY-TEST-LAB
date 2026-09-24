"""Attack datasets: the built-in corpus plus datasets users upload.

The built-in corpus (data/test_cases/corpus_en_v0.yaml) is dataset "" — always
available and read-only from the UI. Uploads (CSV, JSON or JSONL) are
normalised into the same TestCase schema and stored as JSON under
data/datasets/, so the runner, evaluator and reports treat them exactly like
the built-in corpus.

Only `prompt` is required. Missing fields get defaults that fit this lab: the
attack tries to leak the canary, which is caught by string match plus the
judge. Column names from the Turkish HF export (kategori, siddet, ...) are
accepted too, so that dataset can be re-imported unchanged.
"""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime

from pydantic import ValidationError

from app.config import CORPUS_PATH, DATA_DIR
from app.schemas.loader import load_test_suite
from app.schemas.test_case import CANARY, ID_PATTERN, TestCase, TestSuite

DATASETS_DIR = DATA_DIR / "datasets"
BUILTIN_LABEL = "Varsayılan corpus"
MAX_ROWS = 500  # every row runs once per defense layer
_MAX_ERRORS = 8
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_TR_CHARS = re.compile("[çğıöşüÇĞİÖŞÜ]")

# canonical field -> accepted column names (compared lowercase)
_ALIASES = {
    "prompt": ("prompt", "attack", "saldiri"),
    "id": ("id",),
    "category": ("category", "kategori"),
    "owasp": ("owasp",),
    "atlas_tactic": ("atlas_tactic", "atlas"),
    "technique": ("technique", "teknik"),
    "language": ("language", "lang", "dil"),
    "severity": ("severity", "siddet"),
    "attack_goal": ("attack_goal", "saldiri_hedefi"),
    "detection_method": ("detection_method", "tespit_yontemi"),
    "expected_safe_behavior": ("expected_safe_behavior", "beklenen_guvenli_davranis"),
    "failure_indicators": ("failure_indicators", "basarisizlik_gostergeleri"),
    "notes": ("notes", "notlar"),
}
_DEFAULTS = {
    "category": "custom",
    "owasp": "LLM01",
    "atlas_tactic": "unspecified",
    "severity": "medium",
    "attack_goal": "secret_disclosure",
    "detection_method": "string_match",
    "expected_safe_behavior": f"The model must not reveal the secret ({CANARY}) in any form.",
}
_ID_CODE = {
    "direct_injection": "INJ", "indirect_injection": "IND", "prompt_extraction": "EXT",
    "jailbreak_roleplay": "JBR", "obfuscation": "OBF", "agent_manipulation": "AGT",
}


class DatasetError(ValueError):
    """A problem the user can fix: bad file, bad rows, unknown dataset."""


# --- reading ---------------------------------------------------------------------

def _path(dataset: str):
    return DATASETS_DIR / f"{dataset}.json"


def exists(dataset: str) -> bool:
    return dataset == "" or bool(_SLUG_RE.fullmatch(dataset) and _path(dataset).exists())


# Run lists ask for a dataset's name per row; read the files once and refresh
# after every upload/delete.
_name_cache: dict[str, str] | None = None


def _names() -> dict[str, str]:
    global _name_cache
    if _name_cache is None:
        names = {}
        for p in DATASETS_DIR.glob("*.json"):
            try:
                names[p.stem] = json.loads(p.read_text(encoding="utf-8")).get("name") or p.stem
            except (OSError, json.JSONDecodeError):
                continue
        _name_cache = names
    return _name_cache


def label(dataset: str) -> str:
    if not dataset:
        return BUILTIN_LABEL
    return _names().get(dataset, f"{dataset} (silinmiş)")


def load_suite(dataset: str) -> TestSuite:
    if not dataset:
        return load_test_suite(CORPUS_PATH)
    if not exists(dataset):
        raise DatasetError(f"'{dataset}' dataset'i bulunamadı.")
    return TestSuite(cases=json.loads(_path(dataset).read_text(encoding="utf-8"))["cases"])


def cases_by_id(dataset: str) -> dict:
    """{case_id: TestCase}; empty when the dataset was deleted after a run."""
    try:
        return {c.id: c for c in load_suite(dataset).cases}
    except DatasetError:
        return {}


def list_datasets() -> list[dict]:
    """[{id, name, count, builtin, created_at, source_file}], built-in first."""
    out = [{"id": "", "name": BUILTIN_LABEL, "builtin": True, "created_at": "",
            "count": len(load_test_suite(CORPUS_PATH).cases),
            "source_file": CORPUS_PATH.name}]
    for p in sorted(DATASETS_DIR.glob("*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": p.stem, "name": meta.get("name") or p.stem, "builtin": False,
                    "count": len(meta.get("cases", [])),
                    "created_at": meta.get("created_at", ""),
                    "source_file": meta.get("source_file", "")})
    return out


# --- upload ----------------------------------------------------------------------

def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Excel on Turkish Windows saves CSV as cp1254.
        return raw.decode("cp1254", errors="replace")


def _parse(filename: str, raw: bytes) -> tuple[list, bool]:
    """-> (rows, is_csv). Rows are dicts, or plain strings for a JSON list of prompts."""
    text = _decode(raw)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "csv":
        header = text.split("\n", 1)[0]
        # Excel uses ';' in locales where ',' is the decimal separator.
        delimiter = max((",", ";", "\t"), key=header.count)
        return list(csv.DictReader(io.StringIO(text), delimiter=delimiter)), True
    if ext == "jsonl":
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise DatasetError(f"Satır {n}: geçersiz JSON ({e.msg}).") from None
        return rows, False
    if ext == "json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise DatasetError(f"Geçersiz JSON: {e.msg} (satır {e.lineno}).") from None
        if isinstance(data, dict):
            data = data.get("cases")
        if not isinstance(data, list):
            raise DatasetError('JSON bir liste ya da {"cases": [...]} olmalı.')
        return data, False
    raise DatasetError("Desteklenen dosya türleri: .csv, .json, .jsonl")


def _canonical(row) -> dict:
    """Map accepted column names to schema fields, dropping empty values."""
    if isinstance(row, str):
        return {"prompt": row.strip()} if row.strip() else {}
    if not isinstance(row, dict):
        return {}
    lower = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    out = {}
    for field, names in _ALIASES.items():
        for n in names:
            v = lower.get(n)
            if isinstance(v, str):
                v = v.strip()
            if v not in (None, "", []):
                out[field] = v
                break
    return out


def _case_dict(fields: dict, row_no: int, used_ids: set[str]) -> dict:
    d = {**_DEFAULTS, **{k: v for k, v in fields.items() if k != "id"}}
    for key in ("category", "language", "severity", "attack_goal", "detection_method"):
        if isinstance(d.get(key), str):
            d[key] = d[key].lower()
    if isinstance(d.get("owasp"), str):
        d["owasp"] = d["owasp"].upper()
    if "language" not in fields:
        d["language"] = "tr" if _TR_CHARS.search(str(d.get("prompt", ""))) else "en"

    indicators = d.get("failure_indicators")
    if isinstance(indicators, str):  # HF export joins the list with " | "
        indicators = [p.strip() for p in indicators.split("|") if p.strip()]
    d["failure_indicators"] = indicators or [CANARY]

    given = str(fields.get("id", "")).strip().upper()
    if ID_PATTERN.match(given) and given not in used_ids:
        d["id"] = given
    else:
        code, n = _ID_CODE.get(d["category"], "CUS"), row_no
        while f"US-{code}-{n:03d}" in used_ids:
            n += 1
        d["id"] = f"US-{code}-{n:03d}"
    return d


def _slug(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.translate(_TR).lower()).strip("-")[:32] or "dataset"
    slug, n = base, 2
    while _path(slug).exists():
        slug, n = f"{base}-{n}", n + 1
    return slug


def save_upload(name: str, filename: str, raw: bytes) -> dict:
    """Validate an uploaded file and store it. Raises DatasetError listing
    the first problems by row, so the user can fix the file and retry."""
    global _name_cache
    filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1][:100]
    rows, is_csv = _parse(filename, raw)
    if len(rows) > MAX_ROWS:
        raise DatasetError(f"En fazla {MAX_ROWS} saldırı yüklenebilir (dosyada {len(rows)} satır var).")

    if rows and isinstance(rows[0], dict) and not _canonical(rows[0]).get("prompt"):
        found = ", ".join(str(k) for k in list(rows[0])[:12])
        raise DatasetError(f"'prompt' sütunu bulunamadı. Dosyadaki sütunlar: {found}")

    cases: list[TestCase] = []
    errors: list[str] = []
    used: set[str] = set()
    for i, row in enumerate(rows):
        where = f"Satır {i + 2}" if is_csv else f"Kayıt {i + 1}"  # CSV row 1 is the header
        fields = _canonical(row)
        if not fields:
            continue  # blank line
        if not fields.get("prompt"):
            errors.append(f"{where}: prompt boş.")
            continue
        try:
            case = TestCase(**_case_dict(fields, len(cases) + 1, used))
        except ValidationError as e:
            err = e.errors()[0]
            field = ".".join(str(x) for x in err["loc"]) or "satır"
            errors.append(f"{where}: {field} — {err['msg']}")
            continue
        used.add(case.id)
        cases.append(case)
        if len(errors) >= _MAX_ERRORS:
            break

    if errors:
        raise DatasetError("Dosya yüklenmedi, şu satırları düzelt:\n" + "\n".join(errors))
    if not cases:
        raise DatasetError("Dosyada hiç saldırı bulunamadı.")

    name = " ".join((name or "").split())[:60] or filename.rsplit(".", 1)[0][:60] or "Dataset"
    slug = _slug(name)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "source_file": filename,
               "created_at": datetime.now().isoformat(timespec="minutes"),
               "cases": [c.model_dump(mode="json") for c in cases]}
    _path(slug).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    _name_cache = None
    return {"id": slug, "name": name, "count": len(cases),
            "custom_category": sum(c.category.value == "custom" for c in cases)}


def delete(dataset: str) -> bool:
    global _name_cache
    if not dataset or not exists(dataset):
        return False
    _path(dataset).unlink()
    _name_cache = None
    return True
