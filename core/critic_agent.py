"""
Multi-Agent Critique Loop (Generator-Critic-Refiner) — Üretilen kodları şema, niyet ve güvenlik açısından denetleyen eleştirmen ajan.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    """Eleştirmen ajanın karar ve geri bildirim raporu."""
    is_approved: bool
    score: float  # 0.0 - 1.0
    critique_notes: List[str] = field(default_factory=list)
    refined_code: Optional[str] = None


class CodeCriticAgent:
    """
    Generator ajanın yazdığı Python/Pandas veya SQL kodunu statik kurallar ve
    anlamsal denetimle inceleyen, hatalı varsayımları kodu çalıştırmadan önce
    yakalayıp düzelten Eleştirmen (Critic) ve İyileştirici (Refiner) Ajanı.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    @classmethod
    def static_audit(
        cls,
        code: str,
        code_type: str,
        user_message: str,
        available_columns: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Kodu LLM çağrısı yapmadan, milisaniyeler içinde statik kurallarla denetle.
        """
        notes = []
        is_clean = True
        avail_lower = {c.lower(): c for c in available_columns}

        if code_type == "python":
            # 1. result_df atanmış mı?
            if "result_df" not in code:
                notes.append("⚠️ Kod içinde 'result_df' değişkenine atama yapılmamış.")
                is_clean = False

            # 2. Var olmayan kolon kontrolü (regex ile df['...'] veya df[...] taraması)
            col_patterns = re.findall(r"(?:df|result_df)\[['\"]([^'\"]+)['\"]\]", code)
            for cp in col_patterns:
                if cp.lower() not in avail_lower:
                    notes.append(f"❌ '{cp}' adında bir kolon veri setinde bulunmuyor! (Mevcut: {available_columns[:5]}...)")
                    is_clean = False

            # 3. Niyet Kontrolü (Top N / Sıralama)
            q_lower = user_message.lower()
            if any(w in q_lower for w in ("en çok", "en yüksek", "en büyük", "ilk 3", "ilk 5", "top 5")):
                if not any(k in code for k in ("sort_values", "head(", "nlargest", "value_counts")):
                    notes.append("⚠️ Kullanıcı 'en çok/ilk N' sıralaması istedi ancak kodda sıralama/limit bulunmuyor.")
                    is_clean = False

        elif code_type == "sql":
            # 1. SELECT kontrolü
            if not re.search(r"\bSELECT\b", code, re.IGNORECASE):
                notes.append("❌ SQL sorgusu geçerli bir SELECT ifadesi içermiyor.")
                is_clean = False

            # 2. Niyet Kontrolü
            q_lower = user_message.lower()
            if any(w in q_lower for w in ("en çok", "en yüksek", "ilk 5", "top 5")):
                if "ORDER BY" not in code.upper():
                    notes.append("⚠️ Kullanıcı sıralama istedi ancak SQL sorgusunda 'ORDER BY' bulunmuyor.")
                    is_clean = False

        return is_clean, notes

    def evaluate_and_refine(
        self,
        user_message: str,
        generated_code: str,
        code_type: str,
        schema_context: str,
        available_columns: List[str],
    ) -> CriticVerdict:
        """
        Kodu statik ve gerekirse LLM tabanlı denetimden geçirerek nihai onay veya düzeltilmiş kod üretir.
        """
        if not generated_code or not generated_code.strip():
            return CriticVerdict(is_approved=False, score=0.0, critique_notes=["Boş kod üretildi."])

        # 1. Adım: Statik Denetim (Hızlı & Sıfır Maliyet)
        is_clean, static_notes = self.static_audit(
            code=generated_code,
            code_type=code_type,
            user_message=user_message,
            available_columns=available_columns,
        )

        if is_clean:
            logger.info("✅ Code Critic: Kod statik denetimden tam not aldı (Approved).")
            return CriticVerdict(
                is_approved=True,
                score=1.0,
                critique_notes=["Statik şema ve niyet denetimi başarıyla geçildi."],
                refined_code=generated_code,
            )

        # 2. Adım: Refiner Döngüsü (Hata tespit edilirse kodu anında iyileştir)
        logger.warning(f"🔍 Code Critic Eleştirisi: {static_notes}")
        refine_prompt = (
            f"Kullanıcı Sorusu: {user_message}\n\n"
            f"Üretilen {code_type.upper()} Kodu:\n```{code_type}\n{generated_code}\n```\n\n"
            f"Eleştirmen Ajanın (Critic) Bulguları:\n" + "\n".join(static_notes) + "\n\n"
            f"Veri Şeması:\n{schema_context}\n\n"
            f"GÖREVİN:\n"
            f"Eleştirideki eksiklikleri gidererek YALNIZCA düzeltilmiş, çalışan ```{code_type} ... ``` kodunu döndür."
        )

        messages = [
            {"role": "system", "content": "Sen kıdemli bir Kod İnceleme ve Düzeltme (Refiner) uzmanısın. Yalnızca düzeltilmiş kod bloğunu ver."},
            {"role": "user", "content": refine_prompt},
        ]

        try:
            refined_resp = self.llm.chat(messages)
            block_pattern = rf"```{code_type}\s*([\s\S]*?)```"
            matches = re.findall(block_pattern, refined_resp, re.IGNORECASE)
            if matches:
                refined_code = matches[0].strip()
                return CriticVerdict(
                    is_approved=True,
                    score=0.9,
                    critique_notes=static_notes,
                    refined_code=refined_code,
                )
        except Exception as e:
            logger.error(f"Critic refinement LLM call failed: {e}")

        # Eğer refiner başarısız olursa orijinal kodu döndür ama notları ekle
        return CriticVerdict(
            is_approved=False,
            score=0.5,
            critique_notes=static_notes,
            refined_code=generated_code,
        )
