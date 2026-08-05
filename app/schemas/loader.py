"""YAML corpus'unu ve hedef sistemi okuyup Pydantic ile doğrular.

Doğrulama başarısız olursa yükleme burada durur — hatalı senaryo
test runner'a hiç ulaşmaz.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.schemas.test_case import TargetSystem, TestSuite


def load_test_suite(path: str | Path) -> TestSuite:
    """YAML corpus dosyasını okuyup doğrulanmış TestSuite döndürür."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TestSuite(**data)


def load_target_system(path: str | Path) -> TargetSystem:
    """Hedef ("kurban") sistem YAML'ını okuyup doğrular."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TargetSystem(**data)
