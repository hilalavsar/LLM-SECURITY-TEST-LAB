# Kisitlar ve Bulgular

Bu dosya, projenin durustluk degerini artiran gercek muhendislik
gozlemlerini icerir. "Her saldiriyi engeller" gibi bir iddia YOKTUR.

## Donanim / Runtime Bulgulari

### B-01 · CPU offload kacinilmaz (gelistirici workstation'inda)
- **Ortam:** RTX 4050 Laptop, 6141 MiB VRAM. Ryzen 7 7840HS.
- **Gozlem:** Idle durumda arka plan uygulamalari (tarayici, Slack,
  IDE, NVIDIA overlay) ~2.3 GB VRAM tuketiyor. Qwen 2.5 7B Q4_K_M
  yuklendiginde toplam kullanim 5592 MiB'e cikti, yalnizca 329 MiB bos
  kaldi. Model tam GPU'ya sigmadi, bir kismi CPU'ya offload oldu.
- **Etki:** Prompt evaluation ~2.75 tok/sn'e dustu (pure GPU'da 100+
  beklenir). 40 testlik tam kosum ~15-17 dk surer.
- **Sonuc:** Reproduksiyon icin test_run kaydinda VRAM/offload durumu
  ve arka plan yuku not edilmeli. Temiz ortamda (uygulamalar kapali)
  performans ayrica olculecek — baseline vs realistik karsilastirmasi.

## Metodolojik Kisitlar

### M-01 · Dataset boyutu
- (Doldurulacak) Test corpus n=40; fine-tuning icin ayri, genis dataset
  kullaniliyor. Kucuk n'in istatistiksel guven araligina etkisi.

### M-02 · Judge biaslari
- (Doldurulacak) Judge modelin de jailbreak'lenebilirligi.

## Kapsam Disi (bilincli)
- User auth, RBAC, multi-tenant
- Redis/Celery/Kubernetes
- 12B+ modeller (VRAM)
- RAG / tool-calling guvenligi
