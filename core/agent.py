"""
Data Analyst Agent — LLM yanıtlarından kod çıkarma ve çalıştırma orkestratörü.

Akış:
1. Kullanıcı mesajını al
2. Çoklu DataFrame bağlamını (kolonlar, tipler, istatistikler) hazırla
3. Sistem prompt + bağlam + geçmiş + soru → LLM'e gönder
4. LLM yanıtından Python kod bloklarını çıkar
5. Kodu güvenli sandbox'ta çalıştır (tüm datasetler erişilebilir)
6. Hata varsa → LLM'e geri gönder → düzeltilmiş kodu çalıştır (Self-Healing)
7. Sonuçları (grafik, tablo, metin) döndür
"""

import logging
import re
from typing import Optional

import pandas as pd

from core.llm_client import LLMClient
from core.prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    ERROR_FIX_PROMPT,
    build_dataframe_context,
)
from sandbox.executor import SafeExecutor, ExecutionResult

logger = logging.getLogger(__name__)

# LLM sohbetinde izin verilen roller
VALID_ROLES = frozenset({"system", "user", "assistant"})


class DataAnalystAgent:
    """LLM destekli veri analiz ajanı — self-healing özellikli, çoklu dataset destekli."""

    # Sohbet geçmişinde LLM'e gönderilecek maksimum mesaj sayısı
    MAX_HISTORY_MESSAGES = 10

    # Self-healing: maksimum hata düzeltme denemesi
    MAX_FIX_ATTEMPTS = 2

    # Kod bloklarından çıkarılacak anahtar kelimeler (word boundary ile)
    _CODE_KEYWORDS = frozenset({
        "df[", "df.", "df =",
        "px.", "go.", "fig =", "fig.",
        "result_df", "print(",
        "pd.", "np.",
        ".plot", ".describe(", ".info(",
        ".fillna(", ".dropna(", ".groupby(",
        ".merge(", ".value_counts(",
        "datasets[", "df_",
    })

    def __init__(self):
        self.llm = LLMClient()
        self.executor = SafeExecutor(timeout=60)

    # ------------------------------------------------------------------
    # Ortak mesaj oluşturma (DRY — tek kaynak)
    # ------------------------------------------------------------------
    def _build_system_messages(
        self,
        datasets: dict,
        active_key: str,
        chat_history: list[dict],
        history_limit: int = None,
    ) -> list[dict]:
        """System prompt + son N mesajı oluştur (ortak yardımcı)."""
        context = build_dataframe_context(datasets, active_key)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(dataframe_context=context)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        limit = history_limit or self.MAX_HISTORY_MESSAGES
        recent = chat_history[-limit:]
        for msg in recent:
            role = msg.get("role", "")
            # Sadece geçerli rolleri ilet; bozuk kayıtları at
            if role in VALID_ROLES:
                messages.append({"role": role, "content": msg.get("content", "")})

        return messages

    def build_messages(
        self,
        user_message: str,
        datasets: dict,
        active_key: str,
        chat_history: list[dict],
    ) -> list[dict]:
        """
        LLM'e gönderilecek mesaj listesini oluştur.

        Yapı: [system_prompt, ...son_N_mesaj, user_message]
        """
        messages = self._build_system_messages(datasets, active_key, chat_history)
        messages.append({"role": "user", "content": user_message})
        return messages

    # ------------------------------------------------------------------
    # Kod çıkarma
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_code_blocks(text: str) -> list[str]:
        """
        Markdown yanıtından Python kod bloklarını çıkar.

        Desteklenen formatlar:
          ```python  ```Python  ```py  ```
        """
        if not text or not text.strip():
            return []

        # 1) Etiketli bloklar: ```python, ```py, ```Python
        pattern = r"```[Pp](?:ython|y)\s*\n(.*?)```"
        blocks: list[str] = [b.strip() for b in re.findall(pattern, text, re.DOTALL) if b.strip()]

        # Set ile O(1) duplikasyon kontrolü
        seen = set(blocks)

        # 2) Etiketsiz kod blokları — anahtar kelime kontrolü (substring değil, line-based)
        generic_pattern = r"```\s*\n(.*?)```"
        for block in re.findall(generic_pattern, text, re.DOTALL):
            stripped = block.strip()
            if not stripped or stripped in seen:
                continue

            # Line-tabanlı kontrol: herhangi bir satırda geçerli keyword var mı?
            if any(kw in line for line in stripped.splitlines() for kw in DataAnalystAgent._CODE_KEYWORDS):
                blocks.append(stripped)
                seen.add(stripped)

        return blocks

    # ------------------------------------------------------------------
    # Kod çalıştırma + self-healing
    # ------------------------------------------------------------------
    def execute_response_code(
        self,
        response_text: str,
        datasets: dict,
        active_key: str,
    ) -> list[ExecutionResult]:
        """
        LLM yanıtındaki tüm Python kod bloklarını çıkar ve çalıştır.

        Başarısız olanlar için self-healing (fix_and_retry) tetiklenir.
        """
        code_blocks = self._extract_code_blocks(response_text)
        results: list[ExecutionResult] = []

        for code in code_blocks:
            result = self.executor.execute(code, datasets, active_key)

            # Self-healing: hata varsa düzeltmeye çalış
            if not result.success:
                logger.warning("Kod çalıştırma hatası, self-healing başlatılıyor...")
                fixed_result, _ = self.fix_and_retry(
                    failed_code=code,
                    error_message=result.error or "Bilinmeyen hata",
                    datasets=datasets,
                    active_key=active_key,
                    chat_history=[],  # Mevcut bağlam yeterli
                )
                if fixed_result is not None:
                    result = fixed_result

            results.append(result)

        return results

    def fix_and_retry(
        self,
        failed_code: str,
        error_message: str,
        datasets: dict,
        active_key: str,
        chat_history: list[dict],
        attempt: int = 1,
    ) -> tuple[Optional[ExecutionResult], Optional[str]]:
        """
        Self-Healing: Hatalı kodu LLM'e gönder, düzeltilmiş kodu al ve çalıştır.

        Args:
            failed_code: Hata veren Python kodu
            error_message: Hata mesajı
            datasets: Tüm datasetler
            active_key: Aktif dataset anahtarı
            chat_history: Sohbet geçmişi
            attempt: Mevcut deneme sayısı (1-based)

        Returns:
            (ExecutionResult, llm_fix_text) veya (None, None) başarısız olursa
        """
        if attempt > self.MAX_FIX_ATTEMPTS:
            logger.error(f"Self-healing başarısız: {self.MAX_FIX_ATTEMPTS} denemeden sonra vazgeçildi.")
            return None, None

        # Hata düzeltme prompt'unu hazırla
        fix_prompt = ERROR_FIX_PROMPT.format(
            error_message=error_message,
            failed_code=failed_code,
        )

        # Ortak mesaj oluşturucuyu kullan (DRY)
        messages = self._build_system_messages(datasets, active_key, chat_history, history_limit=4)
        messages.append({"role": "user", "content": fix_prompt})

        try:
            fix_response = self.llm.chat(messages)
        except Exception as exc:
            logger.exception(f"LLM hata düzeltme isteği başarısız (deneme {attempt}): {exc}")
            return None, None

        fixed_codes = self._extract_code_blocks(fix_response)

        if not fixed_codes:
            logger.warning(f"LLM kod döndürmedi (deneme {attempt}).")
            return None, fix_response

        # Düzeltilmiş kodu çalıştır
        result = self.executor.execute(fixed_codes[0], datasets, active_key)

        if result.success:
            logger.info(f"Self-healing başarılı (deneme {attempt}).")
            return result, fix_response

        # İlk deneme başarısızsa, tekrar dene (recursive)
        logger.warning(f"kod da hata verdi (deneme {attempt}), tekrar deneniyor...")
        return self.fix_and_retry(
            failed_code=fixed_codes[0],
            error_message=result.error or " kodda bilinmeyen hata",
            datasets=datasets,
            active_key=active_key,
            chat_history=chat_history,
            attempt=attempt + 1,
        )

    def is_llm_available(self) -> bool:
        """llama-server erişilebilir mi?"""
        return self.llm.is_available()