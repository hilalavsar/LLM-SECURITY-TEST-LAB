# Metodoloji

> Iskelet — Hafta 5-6'da doldurulacak.

## Degerlendirme akisi
Ham cevap -> RuleEvaluator -> (REVIEW ise) Judge Panel -> verdict.

## Judge karsilastirma tablosu (hedef)
| Judge | Accuracy | Latency | Maliyet |
|---|---|---|---|
| Prompt-based baseline (kucuk model) | ? | ? | $0 |
| Fine-tuned (Colab QLoRA, senin datan) | ? | ? | $0 |
| (ops.) Uzak API | ? | ? | ~sent |
| Human ground truth | 100% | — | — |

## Anlasma olcumu
- Cohen's Kappa (judge'lar arasi tutarlilik).

## Reproduksiyon
- Her test_run: model, quantization, temperature, seed, top_p,
  sistem prompt hash'i snapshot'lanir.
