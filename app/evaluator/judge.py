"""L3 · Evaluator — Layer 2: Judge Panel.

Kural tabanli katman REVIEW dediginde (kararsiz kaldiginda) devreye
girer. Uc judge'i karsilastirmali calistirma hedefi:
  A. Prompt-based baseline  (kucuk yerel model, sadece prompt)
  B. Fine-tuned             (Colab QLoRA ile egitilmis, senin datan)
  C. (opsiyonel) Uzak API   (kalite tavani icin)

Bilimsel deger: A vs B karsilastirmasi olmadan "fine-tuned daha iyi"
iddiasi gecersiz. Bu yuzden baseline atlanamaz.

Bu dosya HENUZ BOS bir iskelet — Hafta 5-6'da doldurulacak.
"""

from __future__ import annotations
