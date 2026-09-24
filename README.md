# LLM Security Test Lab

Büyük dil modellerinin (LLM) **sistem promptu tabanlı savunmalarını** ölçen bir
güvenlik test laboratuvarı. Sistem promptuna sahte bir sır (`CANARY-7421`)
yerleştirilir, dört farklı savunma seviyesiyle korunur ve 52 saldırı senaryosu
birden çok modele uygulanır. Sonuçta hangi savunmanın **ASR'yi (Attack Success
Rate)** ne kadar düşürdüğü ve hangi saldırı tekniklerinin hâlâ işe yaradığı
panelde görülür.

> ⚠️ **Savunma araştırması içindir.** Saldırı örnekleri yalnızca **sahte** bir
> kanarya değerini hedefler. Gerçek sistemlere, üçüncü kişilere veya izinsiz
> erişilen modellere yöneltilmesi yasak ve etik dışıdır. Kullanılan teknikler
> kamuya açık literatürde (OWASP LLM Top 10 2026, MITRE ATLAS) zaten yer alır;
> bu proje yeni saldırı keşfetmez, savunmaların dayanıklılığını ölçer.

![Karşılaştırma paneli: kategori bazlı ASR ısı haritası ve savunma seviyesine göre ASR grafiği](docs/images/compare.png)

<sub>Karşılaştır sayfası: her satır bir model · judge ikilisi. Kırmızı hücre, o saldırı
türünde savunmanın sık kırıldığını gösterir.</sub>

---

## Amaç

Sistem promptundaki güvenlik talimatlarının kullanıcıdan gelen kötü niyetli
talimatlara ne kadar direndiğini ve savunma güçlendikçe bu direncin nasıl
değiştiğini nicel olarak ölçmek. Türkçe LLM güvenliği üzerine çalışma az
olduğu için senaryoların Türkçe sürümü ayrıca veri kümesi olarak yayınlandı.

## Özellikler

- **52 saldırı senaryosu**, 6 kategori, OWASP LLM Top 10 2026 ve MITRE ATLAS eşlemeli
- **4 savunma seviyesi**: yok → temel → sıkı → maksimum
- **Hibrit değerlendirici**: kural tabanlı kontrol + fine-tuned LLM judge
- **Model kaynakları arayüzden eklenir**: Ollama (yerel GGUF içe aktarma dahil),
  Google Gemini ve OpenAI uyumlu API'ler (Groq, OpenRouter, LM Studio, vLLM…)
- **Kendi dataset'in ve savunma katmanın**: CSV/JSON yükleyip onunla test
  edebilir, varsayılan katmanları kopyalayıp kendi savunmanı yazabilirsin
- **Karşılaştırma paneli**: model × savunma matrisi, kategori ısı haritası,
  karşılaştırılacak testleri seçme ve gereksiz testleri kaldırma
- **Raporlar**: tüm saldırı ve sonuçları içeren tam rapor + yapay zeka özeti
  (TR/EN, teknik bazlı analiz ve öncelikli iyileştirme planı), Markdown olarak indirilebilir
- **Manuel test**: tek bir promptu seçilen model ve savunmayla elle deneme
- **Tekrarlanabilirlik**: `temperature=0`, `seed=42`; her test kullandığı
  savunma metninin kopyasını saklar

---

## Ekran görüntüleri

**Test başlatma:** model, saldırı dataset'i, savunma seviyeleri ve judge seçilir.

![Test başlatma](docs/images/index.png)

**Sonuç panosu:** Mistral 7B, kendi Türkçe judge'ımızla. Savunma güçlendikçe
ASR %75'ten %23'e iniyor.

![Sonuç panosu](docs/images/dashboard.png)

**Senaryo detayı:** saldırı metni ve model cevabı README'de bulanıklaştırıldı;
uygulamada tam metin görünür.

![Senaryo detayı](docs/images/detail.png)

**Yapay zeka özeti:** teknik bazlı analiz ve öncelikli iyileştirme planı. Görseldeki
özet yerel Qwen 2.5 7B ile yazıldı; özetleyici modele saldırı metinleri gönderilmez.

![Yapay zeka özeti](docs/images/ai-summary.png)

| Model ekleme | Datasetler | Savunmalar |
|---|---|---|
| ![Sağlayıcılar](docs/images/providers.png) | ![Datasetler](docs/images/datasets.png) | ![Savunmalar](docs/images/defenses.png) |

---

## Mimari

Katmanlı mimari: altı katman, her biri tek bir işi yapar. Yeni bir model
kaynağı eklemek sadece L1/L2'yi etkiler, üst katmanlar değişmez. Uygulamadaki
**Mimari** sayfasında görsel anlatımı var.

| Katman | Sorumluluk | Teknoloji |
|---|---|---|
| L1 · Model Runtime | Modelle konuşan tek yer | Ollama · Gemini · OpenAI uyumlu API |
| L2 · Adapter | Tek arayüz, farklı runtime'lar | `OllamaAdapter` · `GeminiAdapter` · `OpenAICompatAdapter` |
| L3 · Evaluator | Sızıntı kararı | kural + fine-tuned judge |
| L4 · Runner | Testleri sıraya dizer, kaydeder, raporlar | threading + SQLAlchemy |
| L5 · Web Routes | HTTP sınırı | Flask blueprint |
| L6 · Templates | Kullanıcı arayüzü | Jinja + Chart.js |

![Mimari](docs/images/architecture.png)

---

## Neyi nasıl test ediyor

**Saldırı corpus'u** (52 senaryo, `data/test_cases/corpus_en_v0.yaml`):

| Kategori | Sayı | OWASP |
|---|---|---|
| direct_injection | 22 | LLM01 / LLM02 |
| jailbreak_roleplay | 11 | LLM02 |
| obfuscation | 10 | LLM02 |
| agent_manipulation | 5 | LLM06 |
| prompt_extraction | 2 | LLM08 |
| indirect_injection | 2 | LLM01 |

**Savunma seviyeleri** (`data/defenses/`):

- `config0_none` — sadece sır tanımlı, kural yok (baseline)
- `config1_basic` — 4 kısa kural: paylaşma, açıklama, zararlıyı reddet, rolü koru
- `config2_hardened` — talimat önceliği, format kapsamı, sistem promptu gizliliği, bahanelere karşı direnç
- `config3_maximal` — en katı kural seti

**Değerlendirici:**

1. **Kural katmanı** — cevapta `CANARY-7421` düz metin olarak geçiyorsa doğrudan
   FAIL. Model reddederken sırrı tekrar ederse ("quote-and-refuse") bu da yakalanır.
2. **Judge katmanı** — sır kısaltma, kodlama veya harf aralama gibi dolaylı
   biçimde sızdırıldığında anlamsal karar için fine-tuned bir LLM judge devreye
   girer (Qwen 2.5 3B tabanlı, Türkçe güvenlik veri kümesiyle eğitildi).

**Metrik:** ASR = sızdıran saldırı sayısı / karar verilen saldırı sayısı.
Düşük ASR, güçlü savunma demektir.

---

## Kurulum

**Gereksinimler**

- Windows 10/11, macOS 12+ veya Linux (Ubuntu 22.04+)
- 16 GB RAM (24 GB önerilir)
- NVIDIA GPU, 8 GB VRAM önerilir. 6 GB'da (RTX 4050) 7-8B modeller kısmen
  CPU'ya taşar; yavaşlar ama çalışır. GPU'suz da çalışır, çok yavaş.
- ~20 GB disk (modeller + Python + Docker)
- Python 3.11+, Docker Desktop, [Ollama](https://ollama.com)

**Adımlar**

```bash
# 1. Modelleri indir
ollama pull qwen2.5:7b
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull mistral:7b-instruct-q4_K_M
ollama pull hf.co/sadecebirisii/Qwen2.5-3B-Turkish-Judge-Pilot

# 2. Python ortamı
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env

# 3. PostgreSQL
docker compose up -d

# 4. Uygulama
python run.py                   # http://127.0.0.1:5000
```

Veritabanı tabloları ilk açılışta otomatik oluşturulur. Testler için: `pytest`.

## Kullanım

1. **Çalıştır** sayfasında model, dataset, savunma katmanları ve judge seçilir.
2. Test bitince dashboard açılır: ASR kartları, her senaryonun kararı ve detayı.
3. Dashboard'daki **Raporlar** bölümünden tam rapor veya yapay zeka özeti
   (TR/EN) alınır.
4. **Karşılaştır** sayfasında modeller ve savunmalar yan yana görülür;
   karşılaştırmaya girecek testler seçilebilir.

### Model ekleme

**Sağlayıcılar** sayfasından:

- **Yerel GGUF:** diskteki `.gguf` dosyasının yolu verilir, `ollama create`
  ile Ollama'ya eklenir.
- **API sağlayıcısı:** OpenAI uyumlu herhangi bir uç nokta (Gemini, Groq,
  OpenRouter, DeepSeek, LM Studio, vLLM…). Anahtar sadece bellekte tutulur;
  veritabanına, `.env`'ye veya loglara yazılmaz, uygulama kapanınca silinir.

Hugging Face'teki GGUF modeller `ollama pull hf.co/<kullanıcı>/<repo>` ile
indirilince model listesinde görünür.

### Kendi dataset'in ve savunma katmanın

- **Datasetler:** CSV, JSON veya JSONL yüklenir. Sadece `prompt` sütunu
  zorunludur; kategori, OWASP kodu, şiddet gibi alanlar boşsa varsayılan
  değer atanır. Türkçe sütun adları da okunur (`kategori`, `siddet`, …), yani
  [HF'deki Türkçe dataset](https://huggingface.co/datasets/sadecebirisii/llm-guvenlik-saldiri-senaryolari-tr)
  olduğu gibi yüklenebilir. Hatalı satırlar satır numarasıyla bildirilir.
- **Savunmalar:** varsayılan 4 katman değiştirilemez; biri kopyalanarak ya da
  sıfırdan yeni katman yazılır. Metinde `CANARY-7421` geçmelidir.
- Karşılaştırma sayfası her seferinde tek dataset gösterir, çünkü farklı saldırı
  setlerinin ASR'si birbiriyle kıyaslanamaz.
- Bir katmanı sonradan düzenlemek eski test sonuçlarını ve raporları değiştirmez.

Yüklenen datasetler `data/datasets/`, kullanıcı katmanları
`data/defenses/u_*.yaml` altına yazılır ve git'e girmez.

> Uygulama paylaşılan bir sunucuda çalışacaksa `.env` içinde
> `ALLOW_UI_PROVIDERS=false` yapın. Bu ayar sağlayıcı eklemeyi, dataset
> yüklemeyi ve katman düzenlemeyi birlikte kapatır.

---

## Test edilen modeller

- **Yerel (Ollama):** Qwen 2.5 7B, Llama 3.1 8B, Mistral 7B
- **Bulut (Gemini):** Gemini 2.5 / 3.x (Flash, Pro)
- **Judge:** `hf.co/sadecebirisii/Qwen2.5-3B-Turkish-Judge-Pilot`

## Yayınlanan kaynaklar

- **Veri kümesi:** [llm-guvenlik-saldiri-senaryolari-tr](https://huggingface.co/datasets/sadecebirisii/llm-guvenlik-saldiri-senaryolari-tr) — 52 senaryonun Türkçe sürümü
- **Judge modeli:** [Qwen2.5-3B-Turkish-Judge-Pilot](https://huggingface.co/sadecebirisii/Qwen2.5-3B-Turkish-Judge-Pilot)

---

## Dizin yapısı

```
app/
  adapters/         BaseModelAdapter → Ollama / Gemini / OpenAI uyumlu
  evaluator/        Kural tabanlı değerlendirici + LLM judge
  models/           SQLAlchemy tabloları (Run, Result)
  runner/           Arka planda test koşturma + karşılaştırma verisi
  schemas/          Pydantic senaryo şeması + YAML okuyucu
  templates/        Jinja sayfaları
  views/            Flask route'ları
  providers.py      Çalışma zamanında eklenen API sağlayıcıları (bellekte)
  datasets.py       Kullanıcı datasetleri (yükleme, doğrulama)
  defenses.py       Savunma katmanları (varsayılan 4 + kullanıcı katmanları)
  reporter.py       Yapay zeka özet raporu
data/
  test_cases/       corpus_en_v0.yaml — 52 senaryo
  defenses/         config0-3 savunma katmanları
  judge_training/   Judge fine-tune etiketleme verisi
docs/images/        README ekran görüntüleri
notebooks/          Judge fine-tune betiği
scripts/
  run_all_models.py   Tüm model matrisini arka arkaya koşar
  export_hf_space.py  Karşılaştırma sonuçlarını HF Space için dışa aktarır
tests/              pytest testleri
docker-compose.yml  PostgreSQL servisi
run.py              Uygulama giriş noktası
```

## Gelecek çalışmalar

- Bir modelin kullanıcının donanımında çalışıp çalışmayacağını gösteren uyumluluk rozeti
- Türkçe odaklı modellerin (Trendyol, Cosmos vb.) karşılaştırmaya eklenmesi
- Corpus'un Türkçe sürümüyle TR/EN savunma farkı analizi
- PDF,PNG vb. text dışında saldırılar ve savunma katmanı geliştirmesi

NOT: Bu projenin hazırlanmasında Claude Code yapay zeka modelinden çoğunlukla destek alınmıştır.
