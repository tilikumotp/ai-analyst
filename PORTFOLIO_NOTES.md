# 🎓 AI Data Analyst — Teknik Mimari, CV ve Mülakat Rehberi (Expert Notes)

Bu doküman, projeyi **GitHub, Teknik Sunumlar, CV ve Mülakatlarda** en üst düzeyde (Senior AI / LLM / Data Engineer seviyesinde) anlatabilmeniz için tasarlanmış teknik argümanları ve mimari tasarım kararlarını içerir.

---

## 🏛️ 1. Temel Mimari Savunmalar (Key Technical Pillars)

Mülakatlarda veya proje sunumlarında aşağıdaki ana mimari sütunları vurgulayın:

### 🔹 1. "Deterministic Execution over Generative Guessing" (Üretici Tahmin Yerine Deterministik Yürütme)
> **Problem:** Geleneksel LLM tabanlı veri analiz araçları, büyük veri setleri üzerinde doğrudan ham Pandas/Python kodu üretmeye çalışır. Bu durum; `SettingWithCopyWarning`, bellek şişmesi (OOM), tip uyuşmazlıkları ve en tehlikelisi LLM'in veri toplama (aggregation) adımlarında halüsinasyon görmesine yol açar.
> 
> **Çözümümüz:** Veri işleme katmanını tamamen **Text-to-SQL + SQLite Veritabanı Motoruna** devrettik. Veri çekme, filtreleme, gruplama (`GROUP BY`) ve birleştirme (`JOIN`) işlemleri deterministik ilişkisel veritabanı motorunda gerçekleşir. LLM yalnızca sorgu ve görselleştirme tasarımından sorumludur.

---

### 🔹 2. "Zero-LLM Automated Data Profiling & Health Scoring" (Sıfır-Maliyetli Veri Sağlık Raporu)
> **Problem:** Kullanıcı veri setini yüklediğinde veri kalitesi sorunları (eksik hücreler, aşırı uç değerler, çarpık dağılımlar) henüz fark edilmeden analiz yapılır ve yanlış kararlar alınır.
> 
> **Çözümümüz:** Veri yüklendiği anda çalışan **`DataHealthProfiler`** modülü geliştirdik. LLM çağırmadan (sıfır token maliyetiyle) milisaniyeler içinde eksiklik (%20+ risk), IQR tabanlı aykırı değerler, çarpıklık (skewness) ve genel sağlık skorunu hesaplayarak kullanıcıya anında Veri Sağlık ve Hazırlık Raporu sunar.

---

### 🔹 3. "Dynamic Intent Routing & Specialized Persona Prompting" (Dinamik Niyet Yönlendirme)
> **Problem:** Her analiz tipini (trend analizi, dolandırıcılık/risk denetimi, serbest keşif) tek ve genel bir prompt ile çözmeye çalışmak LLM'in yüzeysel çıktılar üretmesine yol açar.
> 
> **Çözümümüz:** `IntentRouter` mimarisi kurduk. Kullanıcı sorgusunu anlık analiz ederek:
> - **📈 Zaman Serisi & Trend Mimarı:** MoM büyüme, eğim, kırılma noktaları ve mevsimsellik analizi.
> - **🚨 Risk & Anomali Dedektörü:** 3-Sigma, IQR ve şüpheli operasyonel hataları raporlama.
> - **🧭 Stratejik Keşif Danışmanı:** Gizli segmentler, korelasyonlar ve Cross-Sell fırsatları çıkarma.
> - **🎯 Kıdemli Veri Analisti:** Standart kurumsal metrik hesaplamaları.

---

### 🔹 4. "Enterprise Semantic Layer & Metric Store" (Kurumsal Semantik Katman ve Metrik Deposu)
> **Problem:** İş dünyasında "Ciro", "Kar", "Aktif Müşteri" gibi kavramlar farklı departmanlar tarafından farklı yorumlanabilir. LLM'e sadece veri tablosunu vermek, modelin şirket standartlarına uymayan rastgele formüller üretmesine yol açar.
> 
> **Çözümümüz:** Şirketin resmi metriklerini (Resmi Ad, Eş Anlamlılar, İş Tanımı, Teknik SQL Formülü ve Zorunlu Koşullar) içeren bir **Semantik Katman (Semantic Layer)** inşa ettik. Model, "ciro" veya "gelir" dendiğinde kafasına göre değil; resmi tanımlı `SUM(satis_tutari) - SUM(iade_tutari)` formülünü ve `siparis_durumu = 'ONAYLANDI'` şartını kullanır.

---

### 🔹 5. "Hybrid Intent Retrieval & LRU Query Caching" (Hibrit Çağırma ve Önbellekleme)
> **Problem:** Sözlükteki tüm kuralları her promptta LLM'e göndermek bağlamı şişirir. Sadece anahtar kelime eşleştirmek ise kullanıcının dolaylı sorularını (*"ne kadar para kazandık"*) kaçırır.
> 
> **Çözümümüz:** **Hybrid Semantic Retrieval Engine** kurduk. Model, hem tam anahtar kelime/eş anlamlı eşleşmesini hem de Türkçe kök/ek (stem/prefix) ve anlamsal niyet örtüşmesini birlikte puanlar. LRU Query Cache ile sık sorulan sorular milisaniyeler içinde önbellekten döner.

---

### 🔹 6. "Pre-Flight Guardrail Validation (Zero-Cost Early Return)" (Uçuş Öncesi Çelişki Denetimi)
> **Problem:** Kural çatışmalarını LLM SQL ürettikten sonra denetlemek gereksiz token harcar ve yanıtı geciktirir.
> 
> **Çözümümüz:** **Pre-Flight Guardrail** mekanizması geliştirdik. Kullanıcı "İptal edilen siparişlerin cirosu nedir?" gibi zorunlu filtreye (`siparis_durumu = 'ONAYLANDI'`) doğrudan zıt bir soru sorduğunda; sistem **LLM'e hiç gitmeden (sıfır token maliyetiyle)** talebi engeller, yönetişim gerekçesini ve alternatif doğru metriği kullanıcıya sunar.

---

### 🔹 7. "C-Level Executive Summary & Hard-Constraint Data Injection" (Yönetici Özeti Formatı)
> **Problem:** Yöneticiler uzun teknik kod ve açıklamalardan hoşlanmaz; doğrudan net cevaplar ve sayılar ister.
> 
> **Çözümümüz:** Modelin tüm çıktıları katı kurallarla şu 3 başlığa zorlanmıştır:
> 1. **🎯 Yönetici Özeti:** 1-2 cümleyle doğrudan cevap.
> 2. **📊 Temel Bulgular:** Gerçek sayılar, sıralamalar ve pazar paylarıyla maddelenmiş liste.
> 3. **💡 Stratejik İçgörü & Öneri:** Verinin iş dünyasındaki karşılığı ve aksiyon planı.
> Tüm SQL ve teknik detaylar ise `st.expander` içinde derli toplu sunulur.

---

### 🔹 8. "Closed-Loop Self-Healing" (Kapalı Döngü Hata İyileştirme)
> **Problem:** Karmaşık SQL veya Plotly grafik kodlarında sözdizimi hataları analiz sürecini kesintiye uğratır.
> 
> **Çözümümüz:** SQLite ve AST sandbox hataları anında negatif geri besleme ile LLM'e iletilir; model 3 denemeye kadar hatayı kendi kendine düzelterek kullanıcıya kesintisiz deneyim sunar.

---

## 💼 2. CV / Portföy İçin Örnek Proje Açıklaması

### 📄 Proje Başlığı (CV):
**AI Data Analyst — Local Text-to-SQL, Enterprise Semantic Layer & Dynamic Intent Routing Platform**

### 📌 Madde İmleri (Bullet Points):
- **Dinamik Niyet Yönlendirme (Dynamic Intent Routing):** Kullanıcı sorgusunu Trend Mimarı, Risk/Fraud Dedektörü veya Stratejik Keşif Danışmanı modlarına dinamik yönlendiren çoklu-persona mimarisi tasarlandı.
- **Sıfır-LLM Otomatik Veri Profilleme (Data Profiling):** Yükleme anında eksiklik, IQR aykırı değerleri, dağılım çarpıklığı ve veri sağlığı skorunu hesaplayan yüksek performanslı profilleme motoru geliştirildi.
- **Kurumsal Semantik Katman & Metrik Deposu (Semantic Layer):** İş tanımları, SQL formülleri, zorunlu filtreler (`mandatory filters`), sürümleme (`versioning`) ve departman sahipliği (`owner`) içeren dinamik metrik yaşam döngüsü mimarisi kuruldu.
- **Pre-Flight Guardrail Katmanı:** Kurumsal veri standartlarına aykırı sorguları LLM çağrısına gitmeden önce sıfır-maliyetle tespit eden ve erken çıkış (early return) sağlayan koruma katmanı inşa edildi.
- **Deterministik Text-to-SQL & SQLite:** Ham Pandas kodu yerine deterministik ilişkisel veritabanı motoru üzerinde salt-okunur (read-only) SQL sorguları çalıştırılarak sıfır-halüsinasyon veri analitiği sağlandı.
- **C-Level Yönetici Özeti & Şeffaf UI:** Streamlit ve Plotly üzerinde Yönetici Özeti, Temel Bulgular, Stratejik İçgörüler ve Data Governance Kartı sunan tam teşekküllü web platformu geliştirildi.
