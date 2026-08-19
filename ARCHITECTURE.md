# 🧠 AI Data Analyst — Sistem Mimarisi ve Teknik Dokümantasyon

Bu belge, **AI Data Analyst (Zero-Hallucination & Semantic Layer Engine)** uygulamasının uçtan uca sistem mimarisini, veri akışını, bileşenlerini ve çalışma prensiplerini detaylı olarak açıklamaktadır.

---

## 📌 1. Sisteme Genel Bakış (Executive Overview)

**AI Data Analyst**, kullanıcıların yüklediği yapılandırılmış veri setlerini (`.csv`, `.xlsx`, `.sql`, `.sqlite`, SQLite veritabanları) doğal dille sorgulamasına, derinlemesine analiz etmesine ve interaktif görselleştirmeler oluşturmasına olanak tanıyan **yapay zeka destekli kurumsal bir veri analisti platformudur**.

### 🌟 Temel Ayrıştırıcı Özellikler
1. **Sıfır-Halüsinasyon (Zero-Hallucination) Garantisi:** LLM, veri üzerinde işlem yapmadan önce asla sayı veya metrik uyduramaz. 2 Aşamalı ReAct döngüsü ile önce kod/sorgu çalıştırılır, elde edilen **%100 matematiksel gerçek tablodan** yönetici raporu sentezlenir.
2. **Dinamik Çift Motor (Dual-Engine Execution):**
   - `.csv` / `.xlsx` dosyalarında yerel **Python / Pandas** motoru (`df.groupby()`, `df.describe()`, `result_df = ...`) çalışır.
   - `.sql` / `.db` / SQLite dosyalarında **SQL Motoru** (`SELECT ... FROM ...`) çalışır.
3. **Kurumsal Semantik Katman (Semantic Layer & Data Governance):** Şirket içi resmi KPI tanımları (örn: Net Ciro, AOV, Brüt Kar, Churn) YAML tabanlı sözlükten anlamsal olarak çekilir ve sorgulara zorunlu iş kuralları olarak enjekte edilir.
4. **Kapalı Devre Kendi Kendini Onarma (Closed-Loop Self-Healing):** Kod çalıştırma esnasında oluşabilecek sözdizimi veya kolon adı hataları LLM tarafından arka planda otomatik olarak düzeltilir.
5. **Güvenli Kod Çalıştırma Sandbox'ı (`SafeExecutor`):** AST (Soyut Sözdizim Ağacı) düzeyinde güvenlik denetimi ile zararlı Python çağrıları engellenir.

---

## 📐 2. Yüksek Seviye Sistem Mimarisi (Architecture Diagram)

```mermaid
flowchart TD
    User([🧑‍💻 Kullanıcı]) -->|Veri Yükleme: CSV/XLSX/SQL| Ingestion[📂 Veri Yükleme Katmanı]
    User -->|Doğal Dil Sorusu| App[🖥️ Streamlit Arayüzü]

    subgraph Ingestion_Layer [1. Veri Hazırlama & Yükleme Katmanı]
        Ingestion --> CSVLoader[📄 CSVLoader / csv.Sniffer]
        CSVLoader -->|Encoding & Tip Koruması| DataFrames[(📊 Pandas DataFrames)]
        CSVLoader -->|Şema Dönüşümü| SQLite[(💾 SQLite In-Memory DB)]
        DataFrames --> HealthCheck[🩺 Veri Kalite & Sağlık Analizi]
    end

    subgraph Core_Agent [2. Çift Motorlu ReAct Ajan Katmanı]
        App --> IntentRouter[🎯 Intent Router: 4 Farklı Analist Personası]
        IntentRouter --> SemanticLayer[🛡️ Semantik Katman & Pre-Flight Guardrail]
        SemanticLayer --> Phase1[🧠 Aşama 1: Planlama & Kod Üretimi]
        
        Phase1 -->|CSV/Excel ise| PandasEngine[🐍 Pandas / Python Sandbox]
        Phase1 -->|SQL/DB ise| SQLEngine[💾 SQLite Sorgu Motoru]
        
        PandasEngine -->|Hata Oluşursa| PySelfHealing[🩹 Python Self-Healing]
        SQLEngine -->|Hata Oluşursa| SQLSelfHealing[🩹 SQL Self-Healing]
        
        PySelfHealing --> PandasEngine
        SQLSelfHealing --> SQLEngine
        
        PandasEngine --> ResultDF[(📊 result_df Doğrulanmış Tablo)]
        SQLEngine --> ResultDF
    end

    subgraph Synthesis_Layer [3. Doğrulanmış Sentez & Çıktı Katmanı]
        ResultDF --> Phase2[✍️ Aşama 2: Grounded Synthesis]
        Phase2 -->|%100 Gerçek Veriyle Rapor| Report[📋 C-Level Yönetici Özeti & Bulgular]
        PandasEngine --> Plotly[📈 Plotly İnteraktif Grafikler]
        ResultDF --> Export[📥 CSV İndirme Butonu]
    end

    Report --> App
    Plotly --> App
    Export --> App
```

---

## 🧩 3. Temel Mimari Bileşenler

### 3.1. Veri Yükleme ve Ayrıştırma (`ingestion/`)
- **`ingestion/csv_loader.py` (`CSVLoader`):**
  - `csv.Sniffer` ile ayrıştırıcı (virgül `,`, noktalı virgül `;`, tab `\t`, pipe `|`) tespiti.
  - Akıllı Türkçe ve Batı Avrupa karakter kümesi algılama (`utf-8`, `utf-8-sig`, `cp1254`, `iso-8859-9`, `latin-1`).
  - Sayısal doğruluk koruması: Ondalıklı sayıların noktalarını silmeden güvenli `float`/`int` dönüşümü (`4500.00` → `4500.0`).
  - Metin ve model adı bütünlüğü (`1.4 TSI`, `320d`, `Tucson 1.6` gibi ifadeler bozulmaz).
- **`ingestion/db_loader.py` (`DatabaseManager`):**
  - Pandas DataFrame'lerini otomatik optimize in-memory SQLite tablolarına aktarır.
  - Zengin şema enjeksiyonu (`get_schema_context()`): Her kolon için veri tipi, tekil örnek değerler ve `[Min, Max, Ort]` istatistiklerini LLM'e sunar.
  - Hızlı kolon profili ve frekans dağılımı (`get_column_profile`).

### 3.2. Çift Motorlu Yürütme ve ReAct Döngüsü (`core/sql_agent.py`)
Ajan, yüklenen dosya türüne göre iki farklı çalışma motorundan birini seçer:

#### 🔹 A. CSV / Excel Modu (Yerel Python / Pandas):
- Prompt: `CSV_PANDAS_PLANNING_PROMPT`
- LLM doğrudan yerel `df` üzerinde çalışan Python/Pandas kodunu üretir:
  ```python
  result_df = df.groupby('Model')['Units_Sold'].sum().reset_index().sort_values('Units_Sold', ascending=False).head(5)
  fig = px.bar(result_df, x='Model', y='Units_Sold', title='Model Satışları')
  ```
- Kod `SafeExecutor` sandbox'ında güvenli bir şekilde koşturulur.
- Çıktı arayüzde **"🐍 Çalıştırılan Pandas / Python Kodu"** olarak sunulur.

#### 🔹 B. SQL / Veritabanı Modu (SQLite):
- Prompt: `SQL_PLANNING_SYSTEM_PROMPT`
- LLM optimize SQLite sorgusu üretir:
  ```sql
  SELECT Model, SUM(Units_Sold) AS Toplam_Satis
  FROM bmw_sales
  GROUP BY Model
  ORDER BY Toplam_Satis DESC
  LIMIT 5;
  ```
- Sorgu `DatabaseManager` üzerinde deterministik olarak çalıştırılır.
- Çıktı arayüzde **"💾 Çalıştırılan SQL Sorgusu"** olarak sunulur.

---

### 3.3. 2 Aşamalı Sıfır-Halüsinasyon (Zero-Hallucination) Akışı
| Aşama | Sorumluluk | LLM Prompt Davranışı |
| :--- | :--- | :--- |
| **Aşama 1 (Plan & Act)** | Kod / SQL Planlama | *"Sorgu henüz çalışmadığı için ASLA sayı uydurma. Yalnızca planını ve kodunu yaz."* |
| **Yürütme (Execution)** | Deterministik Sandbox | Python/Pandas veya SQLite motoru kodu çalıştırıp `result_df` tablosunu üretir. |
| **Aşama 2 (Grounded Synthesis)** | Yönetici Raporu Yazma | `result_df` metin tablosu LLM'e verilir. LLM **YALNIZCA** bu tablodaki gerçek sayıları kullanarak rapor yazar. |

---

### 3.4. Semantik Katman ve Veri Yönetişimi (`core/knowledge_base.py`)
- **İş Kuralları Kataloğu:** Şirket KPI'ları YAML formatında tanımlanır (`knowledge_base/rules.yaml`).
- **`SemanticRetrievalEngine`:** Kullanıcı sorusuna en uygun resmi iş metriklerini semantik olarak bulur ve sorguya enjekte eder.
- **`GuardrailEngine`:** Pre-Flight denetimi ile şirket politikalarına aykırı veya çelişkili analiz isteklerini erken aşamada yakalar.

### 3.5. Güvenli Kod Çalıştırma Sandbox'ı (`sandbox/executor.py`)
- **AST Tabanlı Güvenlik:** `os`, `sys`, `subprocess`, `socket`, `eval`, `exec`, `open` gibi tehlikeli çağrıları kod çalışmadan önce engeller.
- **Kısıtlı Built-in Ortamı:** Yalnızca güvenli matematik ve veri işleme fonksiyonlarına (`min`, `max`, `sum`, `len`, `range`, `list`, `dict`) izin verir.
- **Zaman Aşımı Koruması (Timeout):** Sonsuz döngüleri engellemek için işlem başına süre limiti uygular.

---

## 📁 4. Proje Dizin Yapısı

```
ai-analyst/
│
├── app.py                      # Streamlit Kullanıcı Arayüzü & Session Orkestrasyonu
├── requirements.txt            # Python Bağımlılıkları
├── .env                        # LLM Bağlantı Ayarları (LM Studio / OpenAI API)
├── ARCHITECTURE.md             # Bu Sistem Mimarisi Dokümantasyonu
│
├── core/                       # Ajan ve LLM Çekirdeği
│   ├── sql_agent.py            # Çift Motorlu ReAct Ajanı & Grounded Synthesis
│   ├── agent.py                # Temel DataAnalystAgent Arayüzü
│   ├── llm_client.py           # LM Studio / OpenAI Uyumlu LLM İstemcisi
│   ├── prompts.py              # Planlama, Sentez ve Persona Prompt Şablonları
│   ├── tools.py                # Tool Calling & Esnek Regex Ayrıştırıcı
│   └── knowledge_base.py       # Semantik Katman, Metrikler & Guardrails
│
├── ingestion/                  # Veri İçe Aktarma & Veritabanı
│   ├── csv_loader.py           # Akıllı CSV/Excel Yükleyici (Sniffer & Encoding)
│   └── db_loader.py            # In-Memory SQLite Yöneticisi & Dinamik Şema
│
├── sandbox/                    # Güvenli Çalıştırma Katmanı
│   └── executor.py             # AST Güvenlikli Python/Pandas Sandbox & Timeout
│
├── knowledge_base/             # Şirket İş Kuralları
│   └── rules.yaml              # Tanımlı Semantik Kurallar ve Metrik Formülleri
│
└── data/                       # Örnek Veri Setleri
    └── ornek_satis_verisi.csv  # Doğrulama ve Demo Veri Seti
```

---

## 🛠️ 5. Teknolojik Yığın (Tech Stack)

- **Frontend & UI:** Streamlit
- **Veri İşleme & Analitik:** Pandas, NumPy
- **Görselleştirme:** Plotly Express & Plotly Graph Objects
- **Veritabanı:** In-Memory SQLite 3
- **Büyük Dil Modeli (LLM):** LM Studio (Yerel Model / Qwen, Llama, Mistral) veya OpenAI Uyumlu API (`/v1/chat/completions`)
- **Güvenlik & İzolasyon:** Python AST Denetimi, Timeout Sinyalleri

---

## 🚀 6. Kurulum ve Çalıştırma

```bash
# 1. Sanal Ortamı Aktif Edin
.\venv\Scripts\activate

# 2. LM Studio'yu Başlatın
# LM Studio uygulamasında Local Server sekmesinden sunucuyu (http://127.0.0.1:1234) başlatın.

# 3. Streamlit Uygulamasını Başlatın
streamlit run app.py
```
