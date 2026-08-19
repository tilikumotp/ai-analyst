# 🧪 AI Data Analyst — Benchmark & Evaluation Scorecard

Bu rapor, sistemin kod çalıştırma doğruluğu (Execution Accuracy) ve halüsinasyonsuz veri sentezi (Groundedness) performansını ölçen otomatik test sonuçlarını içerir.

---

## 📊 Özet Metrikler

| Metrik | Değer | Hedef | Durum |
| :--- | :---: | :---: | :---: |
| **Execution Accuracy (EA)** | **%100.0** | >= 90% | ✅ HEDEF GEÇİLDİ |
| **Groundedness / Faithfulness** | **%100.0** | >= 85% | ✅ SIFIR HALÜSİNASYON |
| **Ortalama Yanıt Süresi** | **6.2 ms** | <= 2500 ms | ✅ ULTRA HIZLI |

---

## 📋 Detaylı Test Sonuçları

| Test ID | Soru | Zorluk | Durum | Süre (ms) | Grounded |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `AUTO_01` | En çok satan ilk 3 araba modelini ve toplam satış adetlerini listele | easy | ✅ PASS | 10.2 ms | ✅ Evet |
| `AUTO_02` | Bölgelere göre toplam satış adetlerini ve pazar paylarını göster | medium | ✅ PASS | 6.8 ms | ✅ Evet |
| `AUTO_03` | 2022 yılından sonraki satışların modeller bazında dağılımı nedir? | medium | ✅ PASS | 4.8 ms | ✅ Evet |
| `RETAIL_01` | Kategori bazında toplam ciro ve ortalama birim fiyatları hesapla | easy | ✅ PASS | 6.0 ms | ✅ Evet |
| `RETAIL_02` | En yüksek satış miktarına sahip ilk 5 şehri ve gelirlerini göster | medium | ✅ PASS | 5.2 ms | ✅ Evet |
| `FINANCE_01` | Aylık satış trendini ve aydan aya değişim oranını göster | hard | ✅ PASS | 4.9 ms | ✅ Evet |
| `FINANCE_02` | Fiyatı 1.000 TL üzerinde olan ve satış adedi 2'den fazla olan işlemleri filtrele | medium | ✅ PASS | 5.3 ms | ✅ Evet |
