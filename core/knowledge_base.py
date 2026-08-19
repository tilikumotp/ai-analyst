"""
Knowledge Base & Semantic Layer — Kurumsal İş Kuralları Deposu, Hibrit Çağırma Motoru,
Önbellekleme (Caching), Yaşam Döngüsü (Lifecycle/Versioning) ve Pre-Flight Guardrail Katmanı.

Bileşenler:
1. BusinessMetric: Versiyonlu, sahiplik ve aktiflik durumlu metrik tanımı
2. KnowledgeBaseManager: CRUD, soft-deprecation ve sürüm yönetimi
3. SemanticRetrievalEngine: Hybrid Search (Exact Keyword + Semantic Intent) + Query-Level Caching
4. GuardrailEngine: Pre-Flight Check (Uçuş Öncesi Erken Çıkış / Çelişki Tespiti)
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_GLOSSARY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "business_glossary.json"
)


def _normalize_text(text: str) -> str:
    """Türkçe karakterleri ve noktalama işaretlerini normalize et."""
    if not text:
        return ""
    text = text.lower()
    mapping = {
        "ç": "c", "ğ": "g", "ı": "i", "i̇": "i",
        "ö": "o", "ş": "s", "ü": "u",
    }
    for tr_char, en_char in mapping.items():
        text = text.replace(tr_char, en_char)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class BusinessMetric:
    """Tek bir kurumsal iş kuralı / metrik tanımı (Yaşam Döngüsü ve Sürüm Destekli)."""
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    business_definition: str = ""
    sql_formula: str = ""
    mandatory_filters: str = ""
    required_columns: List[str] = field(default_factory=list)
    version: str = "1.0"
    is_active: bool = True
    owner: str = "Veri Yönetişimi"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BusinessMetric":
        return cls(
            canonical_name=data.get("canonical_name", ""),
            aliases=data.get("aliases", []),
            business_definition=data.get("business_definition", ""),
            sql_formula=data.get("sql_formula", ""),
            mandatory_filters=data.get("mandatory_filters", ""),
            required_columns=data.get("required_columns", []),
            version=data.get("version", "1.0"),
            is_active=data.get("is_active", True),
            owner=data.get("owner", "Veri Yönetişimi"),
        )


class KnowledgeBaseManager:
    """İş kuralları deposu (Semantic Layer) yöneticisi — Sürüm ve Yaşam Döngüsü Destekli."""

    def __init__(self, file_path: str = DEFAULT_GLOSSARY_PATH):
        self.file_path = file_path
        self._metrics: Dict[str, BusinessMetric] = {}
        self.load()

    def load(self) -> None:
        """JSON dosyasından metrikleri yükle."""
        if not os.path.exists(self.file_path):
            logger.warning(f"Sözlük dosyası bulunamadı: {self.file_path}")
            self._metrics = {}
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._metrics = {
                    item["canonical_name"]: BusinessMetric.from_dict(item)
                    for item in data
                }
            logger.info(f"{len(self._metrics)} iş kuralı yüklendi.")
        except Exception as e:
            logger.error(f"Sözlük yükleme hatası: {e}")
            self._metrics = {}

    def save(self) -> None:
        """Metrikleri JSON dosyasına kaydet."""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        try:
            data = [metric.to_dict() for metric in self._metrics.values()]
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("İş kuralları başarıyla kaydedildi.")
        except Exception as e:
            logger.error(f"Sözlük kaydetme hatası: {e}")

    def list_active_metrics(self) -> List[BusinessMetric]:
        """Yalnızca aktif iş kurallarını listele."""
        return [m for m in self._metrics.values() if m.is_active]

    def list_all_metrics(self) -> List[BusinessMetric]:
        """Tüm metrikleri listele (geçmiş/pasif dahil)."""
        return list(self._metrics.values())

    def get_metric(self, canonical_name: str) -> Optional[BusinessMetric]:
        """İsme göre metrik getir."""
        return self._metrics.get(canonical_name)

    def add_or_update_metric(self, metric: BusinessMetric) -> None:
        """Metrik ekle veya güncelle."""
        self._metrics[metric.canonical_name] = metric
        self.save()

    def deprecate_metric(self, canonical_name: str) -> bool:
        """Kuralı silmeden pasife al (Soft Deprecation — Denetlenebilirlik için)."""
        if canonical_name in self._metrics:
            self._metrics[canonical_name].is_active = False
            self.save()
            return True
        return False

    def delete_metric(self, canonical_name: str) -> bool:
        """Metriği tamamen sil."""
        if canonical_name in self._metrics:
            del self._metrics[canonical_name]
            self.save()
            return True
        return False


class SemanticRetrievalEngine:
    """
    Kullanıcı niyetine göre en alakalı 1-3 iş kuralını çeken Hibrit Çağırma Motoru.
    
    Özellikler:
    - Hybrid Search: Anahtar kelime + Eş anlamlılar + Anlamsal Niyet (Semantic Intent) eşleşmesi
    - Query Caching: Aynı/benzer sorguları önbellekten sıfır gecikmeyle getirme
    - Sadece aktif (`is_active=True`) kuralları dikkate alma
    """

    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb_manager = kb_manager
        self._query_cache: Dict[str, List[BusinessMetric]] = {}

    def clear_cache(self) -> None:
        """Önbelleği temizle."""
        self._query_cache.clear()

    def retrieve(self, user_query: str, top_k: int = 2, threshold: float = 0.25) -> List[BusinessMetric]:
        """
        Kullanıcı sorusuyla eşleşen iş kurallarını getirir (Önbellekli).

        Args:
            user_query: Kullanıcının doğal dil sorusu
            top_k: Döndürülecek maksimum kural sayısı (varsayılan: 2)
            threshold: Minimum kabul edilebilir alaka skoru

        Returns:
            Alakalı BusinessMetric listesi
        """
        norm_query = _normalize_text(user_query)

        # ── 1. Query Caching Kontrolü ──
        if norm_query in self._query_cache:
            logger.debug(f"Retrieval Cache Hit: '{norm_query}'")
            return self._query_cache[norm_query]

        query_tokens = set(norm_query.split())
        scored_metrics: List[Tuple[float, BusinessMetric]] = []

        # Yalnızca aktif kuralları tara (Yaşam Döngüsü Yönetimi)
        active_metrics = self.kb_manager.list_active_metrics()

        for metric in active_metrics:
            score = 0.0

            # 1) Eş Anlamlılar (Aliases) ve Kök/Ön Ek (Stem) Eşleşmesi
            for alias in metric.aliases:
                norm_alias = _normalize_text(alias)
                # Tam kelime/cümle öbeği eşleşmesi
                if re.search(rf"\b{re.escape(norm_alias)}\b", norm_query):
                    score += 1.0
                    break
                elif norm_alias in norm_query:
                    score += 0.85
                    break
                else:
                    alias_tokens = set(norm_alias.split())
                    # Token kümesi veya Türkçe kök (prefix) eşleşmesi (örn: kazandi -> kazan, satislar -> satis)
                    if alias_tokens and alias_tokens.issubset(query_tokens):
                        score += 0.75
                        break
                    
                    # Kök bazlı anlamsal eşleşme (en az 4 karakterli kökler için)
                    stem_matched = False
                    for at in alias_tokens:
                        if len(at) >= 4:
                            stem = at[:5] if len(at) >= 5 else at[:4]
                            if any(qt.startswith(stem) for qt in query_tokens):
                                score += 0.65
                                stem_matched = True
                                break
                    if stem_matched:
                        break

            # 2) Canonical Name kontrolü
            norm_canonical = _normalize_text(metric.canonical_name)
            if norm_canonical in norm_query:
                score += 0.95

            # 3) Anlamsal Tanım (Business Definition) Overlap & Intent Skoru
            norm_def = _normalize_text(metric.business_definition)
            def_tokens = set(norm_def.split())
            intersection = query_tokens.intersection(def_tokens)
            if intersection:
                overlap_ratio = len(intersection) / max(len(query_tokens), 1)
                score += overlap_ratio * 0.45

            if score >= threshold:
                scored_metrics.append((score, metric))

        # En yüksek puana göre sırala ve Top-K al
        scored_metrics.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored_metrics[:top_k]]

        # Önbelleğe kaydet (Maksimum 500 girdi)
        if len(self._query_cache) > 500:
            self._query_cache.clear()
        self._query_cache[norm_query] = results

        return results


class GuardrailEngine:
    """
    Pre-Flight Guardrail Katmanı — LLM çağrısı öncesi maliyetsiz ve hızlı çelişki tespiti.
    
    Kullanıcı talebi ile zorunlu iş kuralları arasındaki mantıksal zıtlıkları denetler.
    """

    # Çelişki yaratan zıt niyet kelimeleri ve ilgili koruma mesajları
    STRICT_CONFLICTS = {
        "siparis_durumu = 'ONAYLANDI'": {
            "keywords": ["iptal", "iade", "onaysiz", "onaylanmamis", "silinen", "reddedilen", "gecersiz"],
            "rule_desc": "yalnızca 'ONAYLANDI' statüsündeki geçerli siparişler",
            "suggestion": "İptal veya iade edilen işlemler için lütfen 'İade Oranı' veya 'İptal Analizi' metriğini talep ediniz.",
        }
    }

    @classmethod
    def pre_flight_check(
        cls, user_query: str, retrieved_metrics: List[BusinessMetric]
    ) -> Tuple[bool, Optional[str]]:
        """
        Uçuş Öncesi Kontrol (Pre-Flight Validation):
        LLM'e hiç gitmeden, sıfır token maliyetiyle katı kural çatışmasını engeller.

        Args:
            user_query: Kullanıcının sorusu
            retrieved_metrics: Seçilen iş kuralları

        Returns:
            (is_blocked: bool, block_reason: Optional[str])
        """
        norm_query = _normalize_text(user_query)

        for metric in retrieved_metrics:
            if not metric.mandatory_filters:
                continue

            for condition, conflict_info in cls.STRICT_CONFLICTS.items():
                if condition in metric.mandatory_filters:
                    for kw in conflict_info["keywords"]:
                        # Kelime sınırıyla tam eşleşme
                        if re.search(rf"\b{re.escape(kw)}\b", norm_query):
                            block_msg = (
                                f"🛡️ **Şirket Veri Yönetişimi / Guardrail Engeli (Pre-Flight):**\n\n"
                                f"Sorunuz `{kw}` ifadesini içeriyor; ancak kurumsal **'{metric.canonical_name}' (v{metric.version})** "
                                f"iş kuralı gereğince ciro/gelir hesaplamaları **{conflict_info['rule_desc']}** için yapılabilir.\n\n"
                                f"💡 *Öneri:* {conflict_info['suggestion']}"
                            )
                            return True, block_msg

        return False, None

    @classmethod
    def inspect(cls, user_query: str, retrieved_metrics: List[BusinessMetric]) -> List[str]:
        """Genel bilgilendirme amaçlı guardrail uyarıları (Bloklamayan yumuşak uyarılar)."""
        warnings: List[str] = []
        is_blocked, msg = cls.pre_flight_check(user_query, retrieved_metrics)
        if is_blocked and msg:
            warnings.append(msg)
        return warnings
