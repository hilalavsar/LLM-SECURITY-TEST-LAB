"""Defense layers: the four built-in configs plus layers users add from the UI.

Built-in layers (data/defenses/config*_en.yaml) are read-only in the UI. User
layers are stored next to them as u_<slug>_en.yaml in the same format, so the
runner and evaluator load both the same way. The "_en" suffix keeps them on
the compare page, which hides legacy Turkish configs that lack it.

Each run stores a snapshot of the defense texts it used (Run.defense_snapshot),
so editing or deleting a user layer never changes what past results and AI
reports say was tested.
"""

from __future__ import annotations

import re
from datetime import datetime

import yaml

from app.config import DEFENSES_DIR
from app.schemas.loader import load_target_system
from app.schemas.test_case import CANARY

USER_PREFIX = "u_"
MAX_LABEL_CHARS = 60
MAX_PROMPT_CHARS = 20_000
_USER_NAME_RE = re.compile(r"u_[a-z0-9-]{1,40}_en")
_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

# Label and description shown for each built-in layer.
BUILTIN: dict[str, tuple[str, str]] = {
    "config0_none_en":     ("Savunma 0 · Yok",        "Baseline · sadece canary tanımlı, hiçbir kural yok. Model saldırıya karşı çıplak."),
    "config1_basic_en":    ("Savunma 1 · Temel",      "SafeAssist persona + 4 kısa kural: kodu paylaşma, talimatı ele verme, zararlı isteklere yardım etme, rol değiştirmeyi reddet."),
    "config2_hardened_en": ("Savunma 2 · Sıkı",       "4 sıkı kural: (1) instruction hierarchy (sistem her zaman kullanıcının üstünde), (2) encoding resistance (base64/morse/spaced vs.), (3) system prompt confidentiality, (4) pretext immunity (\"bu bir test\" işe yaramaz)."),
    "config3_maximal_en":  ("Savunma 3 · Maksimum",   "5 gelişmiş kural: (1) kibar manipülasyon reddi, (2) canary'yi HER formatta koru, (3) sistem prompt sorulursa SAHTE talimat ver (deception), (4) kurgu/roleplay/hipotetik bağışıklığı, (5) yanıt öncesi self-check + 5 standart red formatı."),
}


class DefenseError(ValueError):
    """A problem the user can fix from the form."""


def is_user_layer(name: str) -> bool:
    return bool(_USER_NAME_RE.fullmatch(name or ""))


def _path(name: str):
    return DEFENSES_DIR / f"{name}.yaml"


# Labels are looked up once per results row on busy pages; read the user
# files once and refresh after every save/delete.
_label_cache: dict[str, str] | None = None


def _user_labels() -> dict[str, str]:
    global _label_cache
    if _label_cache is None:
        labels = {}
        for p in DEFENSES_DIR.glob(f"{USER_PREFIX}*.yaml"):
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            labels[p.stem] = data.get("label") or p.stem
        _label_cache = labels
    return _label_cache


def label(name: str) -> str:
    if name in BUILTIN:
        return BUILTIN[name][0]
    return _user_labels().get(name, name)


def text(name: str) -> str | None:
    """Current system-prompt text of a layer, or None if it no longer exists."""
    path = _path(name)
    return load_target_system(path).system_prompt if path.exists() else None


def list_layers() -> list[dict]:
    """[{name, label, desc, system_prompt, builtin}], built-in layers first."""
    out = []
    for p in DEFENSES_DIR.glob("*.yaml"):
        name = p.stem
        builtin = name in BUILTIN
        lbl, desc = BUILTIN.get(name, (label(name), "Kullanıcı tanımlı katman"))
        try:
            system_prompt = load_target_system(p).system_prompt
        except Exception:  # noqa: BLE001 - a hand-broken YAML must not take the page down
            system_prompt = ""
        out.append({"name": name, "label": lbl, "desc": desc,
                    "system_prompt": system_prompt, "builtin": builtin})
    return sorted(out, key=lambda d: (not d["builtin"], d["name"]))


class _BlockDumper(yaml.SafeDumper):
    """Write multi-line strings as | blocks so the saved file stays readable."""


_BlockDumper.add_representer(
    str,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", data, style="|" if "\n" in data else None
    ),
)


def _new_name(label_text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label_text.translate(_TR).lower()).strip("-")[:30] or "katman"
    name, n = f"{USER_PREFIX}{slug}_en", 2
    while _path(name).exists():
        name, n = f"{USER_PREFIX}{slug}-{n}_en", n + 1
    return name


def save(label_text: str, system_prompt: str, name: str = "") -> str:
    """Create a user layer, or overwrite the user layer `name`. Returns its name."""
    global _label_cache
    label_text = " ".join((label_text or "").split())
    system_prompt = (system_prompt or "").replace("\r\n", "\n").strip()
    if not label_text:
        raise DefenseError("Katmana bir ad ver.")
    if len(label_text) > MAX_LABEL_CHARS:
        raise DefenseError(f"Ad en fazla {MAX_LABEL_CHARS} karakter olabilir.")
    if not system_prompt:
        raise DefenseError("Savunma metni boş olamaz.")
    if len(system_prompt) > MAX_PROMPT_CHARS:
        raise DefenseError(f"Savunma metni en fazla {MAX_PROMPT_CHARS} karakter olabilir.")
    # The evaluator scores a run by whether this exact value leaks, so every
    # layer has to hold it — otherwise there is nothing to protect.
    if CANARY not in system_prompt:
        raise DefenseError(
            f"Savunma metninde korunacak gizli değer {CANARY} geçmeli — "
            "değerlendirme bu değerin sızıp sızmadığına bakıyor."
        )
    if name:
        if not is_user_layer(name) or not _path(name).exists():
            raise DefenseError("Sadece kullanıcı katmanları düzenlenebilir.")
    else:
        name = _new_name(label_text)

    header = (f"# Kullanıcı tanımlı savunma katmanı — arayüzden kaydedildi "
              f"{datetime.now():%Y-%m-%d %H:%M}.\n")
    body = yaml.dump(
        {"name": name, "label": label_text, "canary": CANARY,
         "system_prompt": system_prompt + "\n"},
        Dumper=_BlockDumper, allow_unicode=True, sort_keys=False, width=1000,
    )
    _path(name).write_text(header + body, encoding="utf-8")
    _label_cache = None
    return name


def delete(name: str) -> bool:
    global _label_cache
    if not is_user_layer(name) or not _path(name).exists():
        return False
    _path(name).unlink()
    _label_cache = None
    return True
