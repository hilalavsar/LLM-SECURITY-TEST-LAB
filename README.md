# LLM Security Test Lab

Yerel bir LLM'i **prompt injection**, **jailbreak** ve **canary leakage**
senaryolarina karsi test eden; sonuclari metriklerle bir dashboard'da
gosteren kucuk bir guvenlik test laboratuvari.

> Staj projesi · Turkce agirlikli test corpus'u + hybrid evaluator.

## Mimari (ozet)

```
test_cases -> Runner -> Adapter -> Ollama (LLM) -> RuleEvaluator
                                                        |
                                              (REVIEW) Judge Panel
                                                        |
                                             PostgreSQL -> Dashboard
```

Detay: [docs/architecture.md](docs/architecture.md)

## Teknoloji

| Katman | Secim |
|---|---|
| Model runtime | Ollama (yerel) |
| Hedef modeller | Qwen 2.5 7B, Llama 3.1 8B |
| Web | Flask + Jinja + Chart.js |
| Veritabani | PostgreSQL (Docker) |
| Evaluator | Rule-based + Judge panel (hybrid) |

## Hizli baslangic

### 1. On kosullar
- Python 3.11+
- [Ollama](https://ollama.com) kurulu ve calisir durumda
- Docker Desktop (PostgreSQL icin — Hafta 2'den itibaren)

### 2. Modelleri cek
```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 3. Ortami dogrula (smoke test)
Sifir bagimlilik — hemen calisir:
```bash
python scripts/smoke_test.py
```
Bu, Ollama'ya baglanir, bir Turkce prompt gonderir ve kucuk bir canary
sizinti demosu yapar. "Ayarlar (temperature/seed) nerede yapilir?"
sorusunun cevabini script icinde `GEN_OPTIONS` sozlugunde gorursun.

### 4. Gelistirme ortami (sonraki adim)
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env         # sonra .env'i duzenle
```

## Dizin yapisi

```
app/          Flask uygulamasi (katmanlar L2-L6)
  adapters/   BaseModelAdapter + Ollama/Mock
  evaluator/  rules.py + judge.py
  runner/     test runner (CLI)
  models/     SQLAlchemy tablolari
  views/      dashboard ekranlari
data/         test_cases + judge_training
docs/         scope, threat-model, architecture, methodology, limitations
notebooks/    Colab fine-tune defteri
scripts/      smoke_test.py
tests/        unit + integration
```

## Guvenlik notu
Testlerde ASLA gercek sifre, API anahtari veya kisisel veri kullanilmaz.
Yalnizca sahte canary degerleri (or. `CANARY-7421`).
