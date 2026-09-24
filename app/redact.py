"""Secret redaction for text that gets stored or rendered.

Applied to adapter error messages before they reach the database or the UI.
Never applied to model responses — those are attack evidence and must stay
exactly what the model produced.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Reuse the corpus guard's signatures (OpenAI, AWS, GitHub, Slack, PEM) so a
# key format taught to one check is known to both.
from app.schemas.test_case import _REAL_SECRET_PATTERNS

_EXTRA_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),                # Google API key
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),                   # Groq
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/\-]{12,}"),   # echoed Authorization header
]

# Shorter strings would turn ordinary words into *** everywhere.
_MIN_KNOWN_LEN = 8


def redact(text: str | None, known_secrets: Iterable[str] = ()) -> str | None:
    """Replace known secret values and secret-shaped tokens with '***'."""
    if not text:
        return text
    for secret in known_secrets:
        if secret and len(secret) >= _MIN_KNOWN_LEN:
            text = text.replace(secret, "***")
    for pattern in (*_REAL_SECRET_PATTERNS, *_EXTRA_PATTERNS):
        text = pattern.sub("***", text)
    return text
