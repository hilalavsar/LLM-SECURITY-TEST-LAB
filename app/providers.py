"""Çalışma zamanı sağlayıcı kayıt defteri — Arayüz (UI) üzerinden eklenen, OpenAI uyumlu uç noktalar.

Anahtarlar yalnızca ilgili sürecin belleğinde tutulur; veritabanına, .env dosyasına, günlük kayıtlarına veya tarayıcıya hiçbir veri yazılmaz ve uygulama yeniden başlatıldığında tüm sağlayıcı bilgileri silinir. Bu, kişilerin kendi makinelerinde klonlayıp çalıştırdığı bir laboratuvar ortamı için hedeflenen yaşam döngüsüdür: oturum için bir anahtar girilir ve oturum sonunda bu anahtar kaybolur.

Kapsam: tek kullanıcı, yerel makine. Kayıt defteri süreç (process) genelinde geçerlidir; dolayısıyla paylaşımlı veya herkese açık bir dağıtımda her ziyaretçi aynı anahtarları görür ve kullanır. Ayrıca, serbest biçimli `base_url` alanı, herkesin sunucuyu rastgele ana bilgisayarlara (host) istek göndermeye zorlamasına (SSRF) olanak tanır. Uygulamayı bu şekilde herkese açık hale getirmeden önce `ALLOW_UI_PROVIDERS=false` ayarını yapın.

Kasıtlı olarak bir "sağlayıcıyı düzenle" işlemi eklenmemiştir: mevcut bir sağlayıcının `base_url` değerini değiştirmek, kayıtlı anahtarın yeni bir ana bilgisayara gönderilmesine neden olabilir. Bunun yerine sağlayıcıyı silip yeniden ekleyin.

Gemini buraya, `google-generativeai` SDK'sı yerine Google'ın OpenAI uyumlu uç noktası aracılığıyla eklenmiştir; çünkü `genai.configure(api_key=...)` ayarı süreç genelinde (global) geçerlidir ve SDK üzerinden sağlanan sağlayıcıya özel anahtarlar, eşzamanlı işlemler sırasında birbirinin üzerine yazılabilir. İstek başına yapılan standart HTTP çağrılarında ise herhangi bir paylaşılan durum (shared state) söz konusu değildir.
"""

from __future__ import annotations

import ipaddress
import re
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.adapters.openai_compat import OpenAICompatAdapter
from app.config import Settings
from app.redact import redact

PREFIX = "api:"
# OpenRouter alone lists hu ndreds of models; a dropdown that long is useless.
MAX_DISCOVERED_MODELS = 40
# Servers list embedding, image, video, audio and moderation models next to
# chat models (Gemini: text-embedding-004, imagen-*, veo-*; OpenAI: whisper-1,
# dall-e-3, tts-1). They fail on chat/completions, so auto-discovery skips
# them. A name typed into the Modeller field is still accepted as-is.
# Gemini "-live" models only speak the WebSocket Live API (bidiGenerateContent)
# and reject chat/completions with HTTP 400; robotics/computer-use/deep-research
# are agent products on separate APIs.
_NON_CHAT_MARKERS = (
    "embed", "imagen", "veo", "tts", "whisper", "dall-e", "moderation",
    "aqa", "-image", "transcribe", "audio", "lyria",
    "live", "robotics", "computer-use", "deep-research",
)

PRESETS = [
    {"name": "OpenAI",     "base_url": "https://api.openai.com/v1",      "local": False},
    {"name": "Gemini",     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "local": False},
    {"name": "Groq",       "base_url": "https://api.groq.com/openai/v1", "local": False},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",   "local": False},
    {"name": "DeepSeek",   "base_url": "https://api.deepseek.com/v1",    "local": False},
    {"name": "Mistral",    "base_url": "https://api.mistral.ai/v1",      "local": False},
    {"name": "Together",   "base_url": "https://api.together.xyz/v1",    "local": False},
    {"name": "LM Studio",  "base_url": "http://localhost:1234/v1",       "local": True},
    {"name": "vLLM",       "base_url": "http://localhost:8000/v1",       "local": True},
]


class ProviderError(ValueError):
    """A problem the user can fix: bad input, unreachable server, unknown provider."""


@dataclass
class Provider:
    id: str
    name: str
    base_url: str
    # repr=False so an accidental print or log of the object never shows the key.
    api_key: str = field(repr=False)
    models: list[str] = field(default_factory=list)
    hidden_models: int = 0
    added_at: float = field(default_factory=time.time)

    @property
    def masked_key(self) -> str:
        if not self.api_key:
            return "anahtarsız"
        # Only reveal a tail when the key is long enough that 4 chars don't help.
        return "••••" + (self.api_key[-4:] if len(self.api_key) >= 16 else "")

    def model_value(self, model: str) -> str:
        return f"{PREFIX}{self.id}:{model}"


_LOCK = threading.Lock()
_PROVIDERS: dict[str, Provider] = {}
_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


# --- registry -------------------------------------------------------------

def list_providers() -> list[Provider]:
    with _LOCK:
        return sorted(_PROVIDERS.values(), key=lambda p: p.added_at)


def get_provider(provider_id: str) -> Provider | None:
    with _LOCK:
        return _PROVIDERS.get(provider_id)


def remove_provider(provider_id: str) -> bool:
    with _LOCK:
        return _PROVIDERS.pop(provider_id, None) is not None


def clear_providers() -> int:
    with _LOCK:
        count = len(_PROVIDERS)
        _PROVIDERS.clear()
        return count


def add_provider(name: str, base_url: str, api_key: str = "",
                 models_text: str = "") -> Provider:
    """Validate, discover models, and register. Raises ProviderError."""
    name = (name or "").strip()
    if not name:
        raise ProviderError("Sağlayıcıya bir ad ver (örn. Groq).")
    base_url = validate_base_url(base_url)
    api_key = (api_key or "").strip()
    _check_transport(base_url, api_key)

    requested = _parse_models(models_text)
    discovered = _discover(base_url, api_key, required=not requested)

    if requested:
        # Compare against the full listing, not the chat-only subset: typing a
        # name is an explicit choice.
        unknown = [m for m in requested if discovered and m not in discovered]
        if unknown:
            raise ProviderError(
                "Sunucu bu modelleri tanımıyor: " + ", ".join(unknown[:3])
                + ". Adları sunucunun model listesindeki gibi yaz."
            )
        models, hidden = requested, 0
    else:
        chat_models = [m for m in discovered
                       if not any(mark in m.lower() for mark in _NON_CHAT_MARKERS)]
        if not chat_models:
            raise ProviderError(
                "Sunucudan sohbet modeli listelenemedi. Model adlarını elle yaz."
            )
        models = chat_models[:MAX_DISCOVERED_MODELS]
        hidden = len(chat_models) - len(models)

    with _LOCK:
        provider = Provider(_unique_id(name), name, base_url, api_key, models, hidden)
        _PROVIDERS[provider.id] = provider
    return provider


# --- model values -----------------------------------------------------------

def model_choices() -> list[dict]:
    """[{value, label}] for every model of every registered provider."""
    return [
        {"value": p.model_value(m), "label": f"{p.name} · {m}"}
        for p in list_providers() for m in p.models
    ]


def parse_model_value(value: str) -> tuple[str, str] | None:
    """'api:groq:vendor/model:free' -> ('groq', 'vendor/model:free')."""
    if not value.startswith(PREFIX):
        return None
    provider_id, sep, model = value[len(PREFIX):].partition(":")
    if not sep or not provider_id or not model:
        return None
    return provider_id, model


def build_adapter(value: str) -> OpenAICompatAdapter:
    parsed = parse_model_value(value)
    if parsed is None:
        raise ProviderError(f"Geçersiz model değeri: {value}")
    provider_id, model = parsed
    provider = get_provider(provider_id)
    if provider is None:
        raise ProviderError(
            f"'{provider_id}' sağlayıcısı bu oturumda tanımlı değil. Anahtarlar "
            "sadece bellekte tutulur ve uygulama yeniden başlayınca silinir — "
            "Sağlayıcılar sayfasından tekrar ekle."
        )
    return OpenAICompatAdapter(model, provider.base_url, provider.api_key)


def redact_known(text: str | None) -> str | None:
    """Redact every key this process knows about (UI providers + .env Gemini)."""
    with _LOCK:
        keys = [p.api_key for p in _PROVIDERS.values()]
    return redact(text, [*keys, Settings.GEMINI_API_KEY])


# --- validation -------------------------------------------------------------

def validate_base_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ProviderError(
            "Base URL http:// veya https:// ile başlamalı (örn. https://api.openai.com/v1)."
        )
    # Credentials or query strings in the URL would be stored and shown in the
    # UI as plain text. The key has its own field that is never displayed.
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ProviderError(
            "Base URL içinde kullanıcı adı, parola veya ?parametre olmamalı — "
            "anahtarı ayrı alana yaz."
        )
    return url


def _check_transport(base_url: str, api_key: str) -> None:
    parsed = urlparse(base_url)
    if api_key and parsed.scheme == "http" and not _is_loopback(parsed.hostname or ""):
        raise ProviderError(
            "API anahtarı şifresiz http üzerinden uzak bir sunucuya gönderilmez. "
            "https kullan ya da anahtarı boş bırak."
        )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _discover(base_url: str, api_key: str, required: bool) -> list[str]:
    """Fetch /models. Auth failures always raise; other failures only when the
    user gave no model names to fall back on (some servers lack /models)."""
    try:
        return OpenAICompatAdapter.list_models(base_url, api_key)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise ProviderError(
                f"Kimlik doğrulama reddedildi (HTTP {e.code}). API anahtarını kontrol et."
            ) from None
        if required:
            raise ProviderError(
                f"Model listesi alınamadı (HTTP {e.code}). Model adlarını elle yazabilirsin."
            ) from None
    except Exception as e:  # noqa: BLE001
        if required:
            raise ProviderError(
                "Sunucuya ulaşılamadı: " + (redact(str(e), [api_key]) or "")
                + ". Adres doğru mu, sunucu çalışıyor mu?"
            ) from None
    return []


def _parse_models(text: str) -> list[str]:
    seen: list[str] = []
    for item in re.split(r"[,\n]", text or ""):
        item = item.strip()
        if item and item not in seen:
            seen.append(item)
    return seen


def _unique_id(name: str) -> str:
    """Slug used inside model values. Caller holds _LOCK."""
    base = re.sub(r"[^a-z0-9]+", "-", name.translate(_TR).lower()).strip("-") or "provider"
    candidate, n = base, 2
    while candidate in _PROVIDERS:
        candidate, n = f"{base}-{n}", n + 1
    return candidate
