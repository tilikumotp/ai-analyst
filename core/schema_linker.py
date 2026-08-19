"""
Schema Linker — Kullanıcı sorusuna en uygun tabloları ve kolonları dinamik olarak seçen,
büyük veri setlerinde dikkat dağıtıcı (distractor) kolonları eleyen akıllı şema bağlayıcı.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class SchemaLinker:
    """
    Kullanıcı sorusunu analiz ederek veri setindeki en alakalı kolonları belirler
    ve LLM'e sadece bu filtrelenmiş, odaklanmış şemayı (Pruned Schema Context) sunar.
    """

    # Analitik anahtar kelime eşleştirmeleri (Türkçe ve İngilizce)
    SEMANTIC_SYNONYMS = {
        "marka": ["marka", "brand", "manufacturer", "make", "sirket", "company", "araba_markasi"],
        "model": ["model", "series", "seri", "arac_modeli", "urun_adi", "product", "item"],
        "fiyat": ["fiyat", "price", "tutar", "ucret", "cost", "amount", "birim_fiyat"],
        "satis": ["satis", "sales", "units", "units_sold", "adet", "miktar", "quantity", "volume", "toplam_satis"],
        "gelir": ["gelir", "revenue", "ciro", "toplam_gelir", "total_revenue", "kazanc", "income"],
        "tarih": ["tarih", "date", "yil", "year", "ay", "month", "gun", "day", "donem", "period", "quarter"],
        "bolge": ["bolge", "region", "sehir", "city", "ulke", "country", "location", "konum", "alan"],
        "kategori": ["kategori", "category", "segment", "tur", "tip", "type", "grup", "group"],
        "musteri": ["musteri", "customer", "user", "kullanici", "client", "alici"],
        "kar": ["kar", "profit", "margin", "net_kar", "brut_kar"],
    }

    @classmethod
    def _calculate_column_relevance(
        cls,
        col_name: str,
        query: str,
        sample_values: List[Any],
    ) -> float:
        """Bir kolonun kullanıcı sorusuyla anlamsal ve sözcüksel alaka skorunu hesapla."""
        q_lower = query.lower()
        col_lower = col_name.lower().replace("_", " ")
        score = 0.0

        # 1. Birebir veya kısmi kolon adı eşleşmesi
        if col_lower in q_lower:
            score += 5.0
        for part in col_lower.split():
            if len(part) > 2 and part in q_lower:
                score += 3.0

        # 2. Semantik Eş Anlamlılar
        for concept, synonyms in cls.SEMANTIC_SYNONYMS.items():
            if concept in q_lower or any(syn in q_lower for syn in synonyms):
                if any(syn in col_lower for syn in synonyms):
                    score += 4.0

        # 3. Örnek Değer Eşleşmesi (Örn: Soru 'BMW' içeriyorsa ve kolon örneklerinde 'BMW' varsa)
        for s in sample_values:
            s_str = str(s).lower()
            if len(s_str) > 2 and s_str in q_lower:
                score += 6.0
                break

        return score

    @classmethod
    def link_dataframe_columns(
        cls,
        df: pd.DataFrame,
        query: str,
        top_k: int = 8,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Kullanıcı sorusuyla en alakalı kolon listesini döndür.
        Eğer toplam kolon sayısı <= 10 ise tüm kolonları korur.
        """
        all_cols = list(df.columns)
        if len(all_cols) <= top_k:
            return all_cols, {c: 1.0 for c in all_cols}

        scores: Dict[str, float] = {}
        for col in all_cols:
            samples = df[col].dropna().unique()[:5].tolist()
            score = cls._calculate_column_relevance(col, query, samples)
            scores[col] = score

        # Skora göre sırala
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # En yüksek skorlu kolonları seç
        selected_cols: Set[str] = set()
        for col, sc in ranked:
            if sc > 0:
                selected_cols.add(col)
            if len(selected_cols) >= top_k:
                break

        # Eğer hiçbiri eşleşmediyse veya çok azsa ilk kolonları ve sayısal kolonları ekle
        if len(selected_cols) < 3:
            for c in all_cols[:top_k]:
                selected_cols.add(c)

        # Her zaman en az 1 sayısal ve 1 kategorik kolon içermesini garanti et
        numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        for nc in numeric_cols[:2]:
            selected_cols.add(nc)

        ordered_selected = [c for c in all_cols if c in selected_cols]
        logger.info(f"🎯 Schema Linker: {len(all_cols)} kolondan {len(ordered_selected)} tanesi seçildi: {ordered_selected}")
        return ordered_selected, scores

    @classmethod
    def build_focused_dataframe_context(
        cls,
        df: pd.DataFrame,
        query: str,
        file_name: str = "data.csv",
        top_k: int = 10,
    ) -> str:
        """Kullanıcı sorusu için daraltılmış, odaklanmış şema bağlamını üretir."""
        if df is None or df.empty:
            return "Yüklü DataFrame bulunmamaktadır."

        selected_cols, _ = cls.link_dataframe_columns(df, query, top_k=top_k)
        sub_df = df[selected_cols]

        col_info = []
        for col in selected_cols:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            if pd.api.types.is_numeric_dtype(df[col]):
                min_v = df[col].min()
                max_v = df[col].max()
                mean_v = round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else 0
                col_info.append(f"  - `{col}` ({dtype}) [Min: {min_v}, Max: {max_v}, Ort: {mean_v}, Boş: {null_count}]")
            else:
                samples = df[col].dropna().unique()[:6]
                sample_str = ", ".join(f"'{s}'" for s in samples)
                col_info.append(f"  - `{col}` ({dtype}) [Örnek Değerler: {sample_str}, Boş: {null_count}]")

        sample_str = sub_df.head(3).to_string(index=False)
        total_info = f"(Toplam {len(df.columns)} kolondan en alakalı {len(selected_cols)} kolon filtrelendi)"

        return f"""━━━ 📄 Dosya: `{file_name}` ({len(df):,} satır, {len(df.columns)} kolon) {total_info} ━━━
Odaklanmış Kolonlar ve Değer Aralıkları:
{chr(10).join(col_info)}

Örnek Veriler (İlk 3 Satır):
{sample_str}
"""
