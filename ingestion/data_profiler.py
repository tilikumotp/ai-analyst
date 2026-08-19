"""
Data Profiler — Veri Sağlık, Kalite ve Hazırlık Analiz Motoru (Data Profiling).

LLM çağrısı gerektirmeden, yüklenen DataFrame üzerinde milisaniyeler içinde:
- Eksik veri oranları ve kritik kolon tespitleri
- IQR (Interquartile Range) tabanlı aykırı değer (Outlier) tespiti
- Dağılım çarpıklığı (Skewness) analizi
- Kardinalite ve sabit kolon tespiti
- Genel Veri Sağlık Skoru (Data Health Score) ve Yönetici Özeti üretir.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class DataHealthProfiler:
    """Yüklenen verileri otomatik olarak profilleme ve kalite raporu oluşturma motoru."""

    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        """
        DataFrame üzerinde hızlı istatistiksel profilleme yapar.

        Returns:
            Detaylı kalite metrikleri ve yönetici özeti içeren sözlük.
        """
        total_rows, total_cols = df.shape
        duplicate_count = int(df.duplicated().sum())

        # ── 1. Eksik Değer (Missing Values) Analizi ──
        missing_counts = df.isnull().sum()
        missing_cols: List[Dict[str, Any]] = []
        high_missing_cols: List[str] = []

        for col, count in missing_counts.items():
            if count > 0:
                pct = round((count / total_rows) * 100, 1)
                missing_cols.append({
                    "column": col,
                    "count": int(count),
                    "percentage": pct,
                })
                if pct >= 20.0:
                    high_missing_cols.append(col)

        # ── 2. Sayısal Kolonlar: Aykırı Değerler (IQR) & Çarpıklık (Skewness) ──
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_cols: List[Dict[str, Any]] = []
        skewed_cols: List[Dict[str, Any]] = []

        for col in numeric_cols:
            clean_series = df[col].dropna()
            if len(clean_series) < 5:
                continue

            # IQR veya 3-Sigma ile Outlier tespiti
            q25 = clean_series.quantile(0.25)
            q75 = clean_series.quantile(0.75)
            iqr = q75 - q25

            if iqr > 0:
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
            else:
                std = float(clean_series.std())
                mean = float(clean_series.mean())
                if std > 0:
                    lower_bound = mean - 2.5 * std
                    upper_bound = mean + 2.5 * std
                else:
                    lower_bound = float(q25)
                    upper_bound = float(q75)

            outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
            outlier_count = len(outliers)
            if outlier_count > 0:
                outlier_cols.append({
                    "column": col,
                    "outlier_count": outlier_count,
                    "percentage": round((outlier_count / len(clean_series)) * 100, 1),
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2),
                })

            # Çarpıklık (Skewness) tespiti
            try:
                skew = clean_series.skew()
                if abs(skew) >= 1.5:
                    direction = "Sağa Çarpık (Aşırı Yüksek Değerler)" if skew > 0 else "Sola Çarpık (Aşırı Düşük Değerler)"
                    skewed_cols.append({
                        "column": col,
                        "skewness": round(float(skew), 2),
                        "direction": direction,
                    })
            except Exception:
                pass

        # ── 3. Kardinalite & Sabit Kolonlar ──
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        high_cardinality_cols = [
            col for col in df.select_dtypes(include=["object"]).columns
            if (df[col].nunique() / max(total_rows, 1)) > 0.95 and total_rows > 50
        ]

        # ── 4. Genel Sağlık Skoru Hesabı (%0 - %100) ──
        score = 100
        # Eksik veri cezası
        total_missing_cells = df.isnull().sum().sum()
        total_cells = max(total_rows * total_cols, 1)
        missing_cell_ratio = (total_missing_cells / total_cells) * 100
        score -= min(missing_cell_ratio * 2.0, 40)

        # Mükerrer satır cezası
        if total_rows > 0:
            dup_ratio = (duplicate_count / total_rows) * 100
            score -= min(dup_ratio * 1.5, 20)

        # Yüksek eksiklikli kritik kolon cezası
        score -= len(high_missing_cols) * 10
        score = max(int(round(score)), 0)

        if score >= 90:
            status_badge = "🟢 Mükemmel Kalite"
            status_desc = "Veri seti son derece temiz ve analize tam hazır."
        elif score >= 75:
            status_badge = "🟡 İyi Kalite"
            status_desc = "Genel olarak sağlam ancak birkaç eksiklik veya aykırı değer içeriyor."
        elif score >= 50:
            status_badge = "🟠 Dikkat Gerektirir"
            status_desc = "Belirgin eksik veya dengesiz kolonlar mevcut, analiz öncesi temizlik önerilir."
        else:
            status_badge = "🔴 Kritik Kalite Riski"
            status_desc = "Yüksek eksik veri veya mükerrer kayıtlar içeriyor, doğrudan analiz yanıltıcı olabilir."

        # ── 5. Yönetici Bulguları ve Tavsiyeleri ──
        findings = []
        recommendations = []

        if missing_cols:
            for m in missing_cols:
                findings.append(f"⚠️ **Eksik Veri:** `{m['column']}` kolonunun %{m['percentage']}'si boş ({m['count']} satır).")
            recommendations.append("Eksik verileri analizden önce medyan/mod ile doldurmanız (imputation) veya o kolonları filtrelemeniz önerilir.")
        else:
            findings.append("✅ **Tam Veri:** Veri setinde hiçbir eksik (NULL) hücre bulunmamaktadır.")

        if outlier_cols:
            for o in outlier_cols[:3]:
                findings.append(f"📊 **Aykırı Değerler (IQR):** `{o['column']}` kolonunda {o['outlier_count']} adet uç değer tespit edildi.")
            recommendations.append("Aykırı değerler ortalamayı saptırabileceğinden, analizlerde ortalama (mean) yerine medyan metriklerini tercih ediniz.")

        if skewed_cols:
            for s in skewed_cols[:2]:
                findings.append(f"📉 **Dengesiz Dağılım:** `{s['column']}` kolonu {s['direction']} (Skor: {s['skewness']}).")

        if duplicate_count > 0:
            findings.append(f"🔁 **Mükerrer Satırlar:** {duplicate_count} adet yinelenen satır mevcut.")
            recommendations.append("Mükerrer kayıtların kaldırılması analiz doğruluğunu artıracaktır.")

        if constant_cols:
            findings.append(f"⚪ **Sabit Kolonlar:** `{', '.join(constant_cols)}` kolonu tek bir değer taşıyor.")

        if not recommendations:
            recommendations.append("Veri seti herhangi bir ön işleme ihtiyaç duymadan doğrudan deterministik SQL analizine uygundur.")

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "duplicate_count": duplicate_count,
            "health_score": score,
            "status_badge": status_badge,
            "status_desc": status_desc,
            "missing_cols": missing_cols,
            "high_missing_cols": high_missing_cols,
            "outlier_cols": outlier_cols,
            "skewed_cols": skewed_cols,
            "constant_cols": constant_cols,
            "findings": findings,
            "recommendations": recommendations,
        }
