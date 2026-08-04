# Mimari

## Katman Modeli

| Katman | Sorumluluk | Teknoloji |
|---|---|---|
| L6 · Presentation | Dashboard, grafikler | Flask views + Jinja + Chart.js |
| L5 · Application | Routes, config | Flask |
| L4 · Orchestration | Test runner (CLI) | Python, `flask run-tests` |
| L3 · Evaluation | Verdict uretimi | Rule-based + Judge panel |
| L2 · Model Abstraction | Adapter arabirimi | `BaseModelAdapter` |
| L1 · Model Runtime | LLM inference | Ollama (yerel) |
| Persistence | Sonuc saklama | PostgreSQL + SQLAlchemy + Alembic |

## Akis

```
test_cases -> Runner -> Adapter -> Ollama (LLM)
                          |
                          v
                     raw response
                          |
                          v
               RuleEvaluator (L3.1)
                    |        |
             net PASS/FAIL   REVIEW
                    |        |
                    |        v
                    |   Judge Panel (L3.2)
                    |        |
                    v        v
                  test_results (PostgreSQL)
                          |
                          v
                  Flask Dashboard (metrikler)
```

## Onaylanmis Kararlar

- **Runtime:** Ollama (llama.cpp wrapper, VRAM ayni; kurulum hizli).
- **Hedef modeller:** Qwen 2.5 7B (TR dengeli) + Llama 3.1 8B (guclu safety).
- **Framework:** Flask + Jinja (sadelik).
- **DB:** PostgreSQL + Docker.
- **Dil:** Turkce agirlikli + EN alt-kume.
- **Judge:** Hybrid — rule + fine-tuned kucuk model (Colab QLoRA) + baseline.

Detayli gerekce: proje plani (mimari tartisma cikti).
