"""
Semantic Caching Engine — Benzer soruları anlamsal olarak yakalayıp sıfır token maliyetiyle anında döndüren önbellek katmanı.
"""

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


@dataclass
class CachedEntry:
    query_text: str
    dataset_hash: str
    tokens_vector: Dict[str, float]
    executed_code: str
    code_type: str
    result_df: Optional[pd.DataFrame]
    grounded_report: str
    figures: List[go.Figure] = field(default_factory=list)
    lineage_mermaid: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    """
    Kullanıcı sorularını vektörel/anlamsal token temsili ile önbelleğe alan ve
    benzerlik eşiği (Cosine Similarity >= threshold) aşıldığında LLM'e gitmeden
    önceki sonuçları milisaniyeler içinde döndüren kurumsal önbellek motoru.
    """

    def __init__(self, default_threshold: float = 0.90, max_entries: int = 500):
        self.default_threshold = default_threshold
        self.max_entries = max_entries
        self._cache: List[CachedEntry] = []

    # ─────────────────────────────────────────────────────────
    # Vektörel Benzerlik & Tokenizasyon
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def compute_dataset_hash(df: Optional[pd.DataFrame], file_name: str = "") -> str:
        """Veri setinin içeriği ve yapısına göre benzersiz bir SHA-256 hash üret."""
        if df is None or df.empty:
            return hashlib.sha256(file_name.encode("utf-8")).hexdigest()[:16]

        sample_bytes = (
            f"{file_name}_{len(df)}_{list(df.columns)}_{df.head(3).to_json()}"
        ).encode("utf-8")
        return hashlib.sha256(sample_bytes).hexdigest()[:16]

    @classmethod
    def _clean_turkish_word(cls, w: str) -> str:
        """Türkçe kelime kökü ve ek sadeleştirmesi."""
        w = w.lower().strip()
        if len(w) <= 3:
            return w

        # 1. Aşama: Çok harfli çekim eklerini temizle
        w = re.sub(
            r"(?:larını|lerini|larına|lerine|ların|lerin|deki|daki|den|dan|ten|tan|nin|nın|ün|un|in|ın|ları|leri|lar|ler|yı|yi|yu|yü|ya|ye)$",
            "",
            w,
        )

        # 2. Aşama: Kalan tek harfli yönelme/belirtme eklerini temizle (kök en az 3 harf kalacak şekilde)
        if len(w) > 3:
            w = re.sub(r"(?:e|a|ı|i|u|ü)$", "", w)

        return w

    @classmethod
    def _tokenize_and_vectorize(cls, text: str) -> Dict[str, float]:
        """Metni n-gram, kök ve TF tabanlı ağırlıklı vektöre dönüştür."""
        clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        raw_words = [w for w in clean.split() if len(w) > 1 or w.isdigit()]

        # Stopwords (Analiz niyetini etkilemeyen dolgu kelimeleri)
        stopwords = {
            "bu", "şu", "o", "bir", "ve", "ile", "için", "ne", "neler", "nedir",
            "mi", "mı", "mu", "mü", "tane", "olan", "olanlar", "göster",
            "listele", "getir", "bul", "hesapla", "lütfen", "hakkında", "göre",
            "hangisi", "hangileri", "bana", "bize", "ver", "yaz", "çıkar", "ilk"
        }

        filtered = []
        for rw in raw_words:
            if rw not in stopwords:
                cleaned_stem = cls._clean_turkish_word(rw)
                filtered.append(cleaned_stem)

        if not filtered:
            filtered = [cls._clean_turkish_word(w) for w in raw_words]

        # Kelime frekansı ve kökler
        vector: Dict[str, float] = {}
        for w in filtered:
            vector[w] = vector.get(w, 0.0) + 1.5
            # Sayısal değerlere yüksek ağırlık ver (örn: 5, 10, 2023)
            if w.isdigit():
                vector[w] = vector.get(w, 0.0) + 2.0

        # Kelime bigramları (kavramsal bütünlüğü yakalamak için)
        for i in range(len(filtered) - 1):
            bg = f"{filtered[i]}_{filtered[i+1]}"
            vector[bg] = vector.get(bg, 0.0) + 2.0

        # Vektör normalizasyonu (L2 Norm)
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm > 0:
            for k in vector:
                vector[k] /= norm

        return vector

    @staticmethod
    def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """İki seyrek (sparse) vektör arasındaki Cosine Benzerliğini hesapla."""
        intersection = set(vec1.keys()) & set(vec2.keys())
        if not intersection:
            return 0.0
        return sum(vec1[k] * vec2[k] for k in intersection)

    # ─────────────────────────────────────────────────────────
    # Önbellek Sorgulama & Kaydetme
    # ─────────────────────────────────────────────────────────
    def get(
        self,
        user_query: str,
        dataset_hash: str,
        threshold: Optional[float] = None,
    ) -> Optional[CachedEntry]:
        """
        Önbellekte aynı veri setine ait benzer bir soru varsa döndür.
        """
        if not self._cache:
            return None

        th = threshold or self.default_threshold
        query_vec = self._tokenize_and_vectorize(user_query)

        best_entry: Optional[CachedEntry] = None
        best_score = 0.0

        for entry in self._cache:
            if entry.dataset_hash != dataset_hash:
                continue

            sim = self._cosine_similarity(query_vec, entry.tokens_vector)
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if best_entry and best_score >= th:
            best_entry.hit_count += 1
            logger.info(
                f"⚡ Semantic Cache HIT! (Skor: {best_score:.3f}) "
                f"Soru: '{user_query}' ≈ Önbellek: '{best_entry.query_text}'"
            )
            return best_entry

        return None

    def set(
        self,
        user_query: str,
        dataset_hash: str,
        executed_code: str,
        code_type: str,
        result_df: Optional[pd.DataFrame],
        grounded_report: str,
        figures: Optional[List[go.Figure]] = None,
        lineage_mermaid: Optional[str] = None,
    ) -> None:
        """Yeni bir başarılı analiz sonucunu önbelleğe kaydet."""
        if not user_query or not grounded_report:
            return

        # Kapasite kontrolü (FIFO)
        if len(self._cache) >= self.max_entries:
            self._cache.pop(0)

        vec = self._tokenize_and_vectorize(user_query)
        entry = CachedEntry(
            query_text=user_query,
            dataset_hash=dataset_hash,
            tokens_vector=vec,
            executed_code=executed_code,
            code_type=code_type,
            result_df=result_df.copy() if result_df is not None else None,
            grounded_report=grounded_report,
            figures=figures or [],
            lineage_mermaid=lineage_mermaid,
        )
        self._cache.append(entry)
        logger.debug(f"💾 Yeni analiz önbelleğe eklendi: '{user_query}'")

    def clear(self, dataset_hash: Optional[str] = None) -> None:
        """Önbelleği tamamen veya belirli bir veri seti için temizle."""
        if dataset_hash:
            self._cache = [e for e in self._cache if e.dataset_hash != dataset_hash]
        else:
            self._cache.clear()
