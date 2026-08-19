"""
LLM Client — LM Studio ve OpenAI-uyumlu yerel sunucular ile otomatik model algılama desteği.

LM Studio, OpenAI API formatında http://127.0.0.1:1234/v1 endpoint'i sunar.
Bu modül, yüklü olan modeli /v1/models endpoint'inden otomatik olarak algılar ve sorgular.
"""

import logging
import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)


class LLMClient:
    """LM Studio & Yerel LLM sunucuları ile iletişim kuran otomatik model algılayıcı client."""

    def __init__(self):
        base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1").strip()
        # Eğer kullanıcı /v1 eklememişse otomatik ekle
        if not base_url.endswith("/v1") and not "/v1" in base_url:
            base_url = base_url.rstrip("/") + "/v1"

        self.base_url = base_url
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("LLM_API_KEY", "lm-studio"),  # LM Studio herhangi bir API key kabul eder
        )
        self.configured_model = os.getenv("LLM_MODEL", "auto").strip()
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self._cached_model: Optional[str] = None

    def get_active_model(self) -> str:
        """
        LM Studio veya yerel sunucudan yüklü olan aktif modeli otomatik algıla.
        
        Eğer model adı 'auto' ise veya belirtilmemişse, LM Studio'da o anda yüklü
        olan ilk modelin adını (/v1/models) otomatik alır.
        """
        if self.configured_model and self.configured_model.lower() not in ("auto", "default", ""):
            return self.configured_model

        try:
            models = self.client.models.list()
            if models.data and len(models.data) > 0:
                detected_model = models.data[0].id
                self._cached_model = detected_model
                logger.info(f"LM Studio Aktif Model Algılandı: {detected_model}")
                return detected_model
        except Exception as e:
            logger.debug(f"Otomatik model algılama hatası: {e}")

        return self._cached_model or "local-model"

    def chat(self, messages: list[dict]) -> str:
        """Senkron chat completion — tam yanıt döndürür."""
        model = self.get_active_model()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_stream(self, messages: list[dict]):
        """
        Streaming chat completion — metin parçalarını yield eder.
        """
        model = self.get_active_model()
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def is_available(self) -> bool:
        """LM Studio sunucusunun çalışıp çalışmadığını kontrol et."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def get_model_info(self) -> str:
        """Arayüzde göstermek için aktif model adını döndür."""
        return self.get_active_model()
