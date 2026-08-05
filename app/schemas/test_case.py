"""Test corpus şeması ve doğrulama kuralları.

Her saldırı senaryosu bu modele karşı doğrulanır. Amaç: elle yazılan
YAML corpus'unda küçük hataları (yazım hatası, eksik alan, tekrar eden ID,
yanlışlıkla konan gerçek secret) çalıştırma anında değil, YÜKLEME anında
yakalamak.

Kapsanan OWASP LLM Top 10 (2026) maddeleri: LLM01, LLM02, LLM08.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator, model_validator

# Projedeki tek sahte canary değeri. target_system.yaml ile aynı olmalı.
# İleride config'ten okunacak; şimdilik sabit referans.
CANARY = "CANARY-7421"

# id deseni: <DİL>-<KATEGORİ>-<NUMARA>  →  örn. TR-INJ-001
ID_PATTERN = re.compile(r"^[A-Z]{2}-[A-Z]{3}-\d{3}$")

# Gerçek-görünümlü secret imzaları. Bunlardan biri eşleşirse senaryo reddedilir.
# (Sadece sahte CANARY serbest — bu desenlerin hiçbirine uymaz.)
_REAL_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),              # OpenAI API key (sk-... / sk-proj-...)
    re.compile(r"AKIA[0-9A-Z]{16}"),                   # AWS access key id
    re.compile(r"ghp_[A-Za-z0-9]{36}"),                # GitHub personal token
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),       # Slack token
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), # PEM özel anahtar
]


def _contains_real_secret(text: str) -> bool:
    return any(p.search(text) for p in _REAL_SECRET_PATTERNS)


# --- Kontrollü kelime dağarcığı (enum'lar) -------------------------------
# str tabanlı Enum: hem tip güvenliği hem YAML'de düz string olarak yazılır.

from enum import Enum


class OwaspCategory(str, Enum):
    LLM01 = "LLM01"  # Prompt Injection
    LLM02 = "LLM02"  # Sensitive Information Disclosure
    LLM08 = "LLM08"  # Hidden Context Exposure


class AttackCategory(str, Enum):
    DIRECT_INJECTION = "direct_injection"
    PROMPT_EXTRACTION = "prompt_extraction"
    JAILBREAK_ROLEPLAY = "jailbreak_roleplay"
    OBFUSCATION = "obfuscation"
    INDIRECT_INJECTION = "indirect_injection"


class Language(str, Enum):
    TR = "tr"
    EN = "en"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackGoal(str, Enum):
    # Note: "canary" is the mechanism (a planted marker secret); the attack goal
    # is framed with the OWASP term. secret_disclosure -> OWASP LLM02.
    SECRET_DISCLOSURE = "secret_disclosure"
    SYSTEM_PROMPT_DISCLOSURE = "system_prompt_disclosure"
    POLICY_BYPASS = "policy_bypass"


class DetectionMethod(str, Enum):
    STRING_MATCH = "string_match"  # rule-based: failure_indicators aranır
    JUDGE = "judge"                # semantik: LLM judge değerlendirir


# --- Tek bir test senaryosu ----------------------------------------------


class TestCase(BaseModel):
    id: str = Field(description="Benzersiz kimlik, TR-INJ-001 formatında")
    owasp: OwaspCategory
    atlas_tactic: str = Field(description="MITRE ATLAS taktiği, örn. AML.TA0004")
    category: AttackCategory
    technique: str | None = Field(default=None, description="Alt-teknik, opsiyonel")
    language: Language
    severity: Severity
    attack_goal: AttackGoal
    detection_method: DetectionMethod = Field(
        description="Bu senaryo nasıl değerlendirilecek: string_match (rule) | judge (semantik)"
    )
    prompt: str = Field(min_length=1, description="Modele gönderilecek saldırı metni")
    expected_safe_behavior: str = Field(
        min_length=1, description="Güvenli davranışın açıklaması (judge için, semantik)"
    )
    failure_indicators: list[str] = Field(
        default_factory=list,
        description="Cevapta görünürse FAIL sayılan ifadeler (rule-based için)",
    )
    notes: str | None = None

    # 2. ID format deseni: TR-INJ-001 gibi olmalı.
    @field_validator("id")
    @classmethod
    def _check_id_format(cls, v: str) -> str:
        if not ID_PATTERN.match(v):
            raise ValueError(f"id '{v}' beklenen desene uymuyor (örn. TR-INJ-001)")
        return v

    # 3. Sadece sahte canary: prompt + indicators içinde gerçek secret olmasın.
    @field_validator("prompt")
    @classmethod
    def _prompt_no_real_secret(cls, v: str) -> str:
        if _contains_real_secret(v):
            raise ValueError("prompt gerçek-görünümlü bir secret içeriyor — yasak")
        return v

    @field_validator("failure_indicators")
    @classmethod
    def _indicators_no_real_secret(cls, v: list[str]) -> list[str]:
        for item in v:
            if _contains_real_secret(item):
                raise ValueError(f"failure_indicators gerçek secret içeriyor: {item!r}")
        return v

    # 4. string_match ile değerlendirilecekse en az bir indicator zorunlu.
    #    (judge ile değerlendirilenler indicator olmadan geçebilir — örn. akrostiş.)
    @model_validator(mode="after")
    def _string_match_needs_indicator(self) -> "TestCase":
        if self.detection_method == DetectionMethod.STRING_MATCH and not self.failure_indicators:
            raise ValueError(
                f"{self.id}: detection_method=string_match ise failure_indicators boş olamaz"
            )
        return self


# --- Corpus'un tamamı (koleksiyon) ---------------------------------------


class TestSuite(BaseModel):
    """Bütün senaryoların kabı. Koleksiyon-seviyesi kuralları burada uygulanır."""

    cases: list[TestCase]

    # 1. Benzersiz ID: aynı id iki senaryoda olamaz.
    @model_validator(mode="after")
    def _unique_ids(self) -> "TestSuite":
        ids = [c.id for c in self.cases]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"Tekrar eden ID'ler var: {dupes}")
        return self


# --- Saldırdığımız sahte hedef ("kurban") sistem -------------------------


class TargetSystem(BaseModel):
    """Saldırıların hedeflediği sahte asistan (sistem prompt + canary)."""

    name: str
    system_prompt: str = Field(min_length=1)
    canary: str

    # canary, sistem prompt içinde geçmeli — yoksa canary testleri anlamsız.
    @model_validator(mode="after")
    def _canary_in_prompt(self) -> "TargetSystem":
        if self.canary not in self.system_prompt:
            raise ValueError("canary değeri system_prompt içinde geçmiyor")
        return self
