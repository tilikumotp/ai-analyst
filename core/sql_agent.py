"""
SQL ReAct Agent — Text-to-SQL, Semantic Layer, 2 Aşamalı Sıfır-Halüsinasyon (Zero-Hallucination) & Self-Healing Orkestratörü.

2 Aşamalı ReAct Akışı (2-Phase Architecture):
1. Aşama (Plan & Act):
   - IntentRouter ile kullanıcı analist personasını belirler.
   - Semantik Katman (Semantic Layer) kurallarını getirir.
   - Pre-Flight Guardrail ile çelişkileri engeller.
   - LLM veritabanı şemasına göre deterministik SQL / CSV inceleme sorgusunu planlar (Sayı uydurmadan).
   - SQLite motorunda SQL çalıştırılır (Hata olursa Self-Healing döngüsü işletilir).
   - result_df elde edilir.
2. Aşama (Grounded Synthesis - Gözlem -> Doğrulanmış Rapor):
   - Gerçek sorgu sonuç tablosu (%100 matematiksel gerçeklik) LLM'e verilir.
   - LLM YALNIZCA bu gerçek verilere dayanarak Yönetici Özeti, Temel Bulgular ve Stratejik Önerileri yazar.
   - Opsiyonel Plotly görselleştirmesi result_df üzerinde çalıştırılır.
"""

import logging
import sqlite3
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from core.knowledge_base import (
    BusinessMetric,
    GuardrailEngine,
    KnowledgeBaseManager,
    SemanticRetrievalEngine,
)
from core.llm_client import LLMClient
from core.prompts import (
    ANOMALY_PLANNING_PROMPT,
    ANOMALY_SYNTHESIS_PROMPT,
    AUTONOMOUS_EXPLORER_PLANNING_PROMPT,
    EXPLORATION_SYNTHESIS_PROMPT,
    GROUNDED_SYNTHESIS_PROMPT,
    SQL_ERROR_FIX_PROMPT,
    SQL_PLANNING_SYSTEM_PROMPT,
    TREND_PLANNING_PROMPT,
    TREND_SYNTHESIS_PROMPT,
)
from core.tools import ToolCall, ToolParser
from ingestion.db_loader import DatabaseManager
from sandbox.executor import ExecutionResult, SafeExecutor

logger = logging.getLogger(__name__)


class AnalyticsIntent(Enum):
    """Kullanıcı analiz niyetleri."""
    GENERAL = "general"
    TREND = "trend"
    ANOMALY = "anomaly"
    EXPLORATION = "exploration"


class IntentRouter:
    """Kullanıcı sorusuna göre en uygun analist personasını ve prompt şablonunu belirler."""

    TREND_KEYWORDS = {
        "trend", "zaman", "aylık", "aylara göre", "yıllık", "yıllara göre", "günlük",
        "mevsimsel", "mevsimsellik", "büyüme", "daralma", "mom", "yoy", "çeyrek",
        "zaman serisi", "artış", "azalış", "kırılma", "tarih"
    }

    ANOMALY_KEYWORDS = {
        "anomali", "aykırı", "outlier", "risk", "şüpheli", "fraud", "hata",
        "aşırı", "beklenmedik", "uç değer", "uç", "sapma", "güvenlik"
    }

    EXPLORATION_KEYWORDS = {
        "ilginç", "keşfet", "keşif", "fırsat", "özetle", "bana ne önerirsin",
        "insight", "gizli", "korelasyon", "segment", "cross-sell", "ilişki",
        "fikir ver", "ne görüyorsun", "derinlemesine"
    }

    @classmethod
    def route(cls, user_query: str) -> Tuple[AnalyticsIntent, str]:
        """
        Kullanıcı niyetini skor tabanlı çoklu anahtar kelime eşleşmesiyle belirler.
        """
        q_lower = user_query.lower()

        scores = {
            AnalyticsIntent.ANOMALY: 0,
            AnalyticsIntent.TREND: 0,
            AnalyticsIntent.EXPLORATION: 0,
        }

        for kw in cls.ANOMALY_KEYWORDS:
            if kw in q_lower:
                scores[AnalyticsIntent.ANOMALY] += 2 if len(kw) > 5 else 1

        for kw in cls.EXPLORATION_KEYWORDS:
            if kw in q_lower:
                scores[AnalyticsIntent.EXPLORATION] += 2 if len(kw) > 5 else 1

        for kw in cls.TREND_KEYWORDS:
            if kw in q_lower:
                scores[AnalyticsIntent.TREND] += 2 if len(kw) > 5 else 1

        max_intent = max(scores, key=lambda k: scores[k])
        if scores[max_intent] > 0:
            if max_intent == AnalyticsIntent.ANOMALY:
                return AnalyticsIntent.ANOMALY, "🚨 Risk & Anomali Dedektörü"
            elif max_intent == AnalyticsIntent.EXPLORATION:
                return AnalyticsIntent.EXPLORATION, "🧭 Stratejik Keşif Danışmanı"
            elif max_intent == AnalyticsIntent.TREND:
                return AnalyticsIntent.TREND, "📈 Zaman Serisi & Trend Mimarı"

        return AnalyticsIntent.GENERAL, "🎯 Kıdemli Veri Analisti"


@dataclass
class AgentStepResult:
    """ReAct Ajanının bir çalıştırma sonucu."""
    explanation: str = ""
    sql_query: Optional[str] = None
    executed_code: Optional[str] = None
    code_type: str = "python"  # "python" veya "sql"
    result_df: Optional[pd.DataFrame] = None
    figures: List[go.Figure] = field(default_factory=list)
    plot_code: Optional[str] = None
    stdout: str = ""
    error: Optional[str] = None
    healing_notes: List[str] = field(default_factory=list)
    applied_metrics: List[BusinessMetric] = field(default_factory=list)
    guardrail_warnings: List[str] = field(default_factory=list)
    total_sql_retries: int = 0
    detected_intent: str = "🎯 Kıdemli Veri Analisti"

    @property
    def has_figure(self) -> bool:
        return len(self.figures) > 0

    @property
    def has_table(self) -> bool:
        return self.result_df is not None and not self.result_df.empty

    @property
    def has_error(self) -> bool:
        return self.error is not None

    @property
    def has_applied_metrics(self) -> bool:
        return len(self.applied_metrics) > 0


class SQLReActAgent:
    """CSV/Pandas ve SQL Veritabanı Modlu, Semantik Katman ve 2 Aşamalı Sıfır-Halüsinasyon Analist Ajanı."""

    MAX_SQL_RETRIES = 3
    MAX_PLOT_RETRIES = 2
    MAX_HISTORY_MESSAGES = 10

    def __init__(
        self,
        db_manager: DatabaseManager,
        kb_manager: Optional[KnowledgeBaseManager] = None,
    ):
        self.db_manager = db_manager
        self.kb_manager = kb_manager or KnowledgeBaseManager()
        self.retrieval_engine = SemanticRetrievalEngine(self.kb_manager)
        self.guardrail_engine = GuardrailEngine()
        self.llm = LLMClient()
        self.executor = SafeExecutor(timeout=60)

    # ─────────────────────────────────────────────────────────
    # Aşama 1: Planlama Mesajları (Planning Prompt)
    # ─────────────────────────────────────────────────────────
    def build_planning_messages(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        relevant_metrics: Optional[List[BusinessMetric]] = None,
        datasets: Optional[Dict[str, Any]] = None,
        active_key: Optional[str] = None,
        mode: str = "python",
    ) -> Tuple[List[Dict[str, str]], str, AnalyticsIntent]:
        """Aşama 1 için niyet ve dosya tipine göre planlama mesajlarını derler."""
        from core.prompts import CSV_PANDAS_PLANNING_PROMPT, build_dataframe_context

        intent, persona_label = IntentRouter.route(user_message)

        if mode == "python" and datasets and active_key and active_key in datasets:
            active_df = datasets[active_key]["df"]
            meta = datasets[active_key].get("metadata", {})
            file_name = meta.get("dosya_adi", active_key)
            df_ctx = build_dataframe_context(active_df, file_name)
            system_prompt = CSV_PANDAS_PLANNING_PROMPT.format(dataframe_context=df_ctx)
        else:
            schema_context = self.db_manager.get_schema_context()
            if relevant_metrics:
                rules_parts = [
                    f"📌 **{m.canonical_name}** (v{m.version})\n• Tanım: {m.business_definition}\n• Formül: `{m.sql_formula}`"
                    for m in relevant_metrics
                ]
                semantic_context = "\n\n".join(rules_parts)
            else:
                semantic_context = "Genel analitik mantığını kullan."

            if intent == AnalyticsIntent.TREND:
                template = TREND_PLANNING_PROMPT
            elif intent == AnalyticsIntent.ANOMALY:
                template = ANOMALY_PLANNING_PROMPT
            elif intent == AnalyticsIntent.EXPLORATION:
                template = AUTONOMOUS_EXPLORER_PLANNING_PROMPT
            else:
                template = SQL_PLANNING_SYSTEM_PROMPT

            system_prompt = template.format(
                semantic_rules_context=semantic_context,
                schema_context=schema_context,
            )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        recent_history = chat_history[-self.MAX_HISTORY_MESSAGES:]
        for msg in recent_history:
            role = msg.get("role", "user")
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": msg.get("content", "")})

        messages.append({"role": "user", "content": user_message})
        return messages, persona_label, intent

    # ─────────────────────────────────────────────────────────
    # 2 Aşamalı ReAct Döngüsü (CSV Pandas / SQL Dual Mode)
    # ─────────────────────────────────────────────────────────
    def execute_react_cycle(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        datasets: Optional[Dict[str, Any]] = None,
        active_key: Optional[str] = None,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[AgentStepResult, str]:
        """
        ReAct Döngüsü:
        - CSV/Excel dosyası yüklendiğinde doğrudan Python / Pandas komutları çalıştırır.
        - SQL veritabanı bağlandığında SQL sorguları çalıştırır.
        """
        step = AgentStepResult()

        # Dosya tipine göre çalışma modunu belirle (CSV/Excel -> python, SQL -> sql)
        mode = "python"
        if datasets and active_key and active_key in datasets:
            meta = datasets[active_key].get("metadata", {})
            file_type = meta.get("dosya_tipi", "").upper()
            if file_type in ("SQL", "SQLITE", "DATABASE"):
                mode = "sql"
            else:
                mode = "python"
        elif self.db_manager.get_table_names():
            # Eğer doğrudan veritabanı tabloları varsa
            mode = "sql"

        step.code_type = mode

        # ── 1. Dinamik Niyet Yönlendirme (Intent Routing) ──
        intent, persona_label = IntentRouter.route(user_message)
        step.detected_intent = persona_label

        if status_callback:
            status_callback("intent", f"{persona_label} devreye girdi...")

        # ── 2. Semantik Katman: Akıllı Kural Çağırma (Retrieval Engine) ──
        if status_callback:
            status_callback("retrieval", "📚 Şirket iş kuralları ve semantik sözlük taranıyor...")

        relevant_metrics = self.retrieval_engine.retrieve(user_message, top_k=2)
        step.applied_metrics = relevant_metrics

        # ── 3. Pre-Flight Guardrail Denetimi ──
        is_blocked, block_msg = self.guardrail_engine.pre_flight_check(user_message, relevant_metrics)
        if is_blocked and block_msg:
            step.explanation = block_msg
            step.guardrail_warnings = [block_msg]
            if status_callback:
                status_callback("guardrail", "🛡️ Pre-Flight Guardrail: Kurumsal kural çatışması engellendi.")
            return step, block_msg

        if status_callback:
            mode_desc = "Python / Pandas" if mode == "python" else "SQL"
            status_callback("reasoning", f"🧠 {persona_label} {mode_desc} analizini planlıyor...")

        # ── 4. Aşama 1: Planlama ve Kod Üretimi ──
        messages, _, current_intent = self.build_planning_messages(
            user_message=user_message,
            chat_history=chat_history,
            relevant_metrics=relevant_metrics,
            datasets=datasets,
            active_key=active_key,
            mode=mode,
        )
        raw_planning_response = self.llm.chat(messages)

        # ── 5. Kod Çalıştırma (Act) ──
        if mode == "python" and datasets and active_key:
            # CSV / Pandas Modu: Python kod bloğunu çıkar ve sandbox'ta çalıştır
            py_blocks = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", raw_planning_response, re.IGNORECASE)
            code_to_run = py_blocks[0].strip() if py_blocks else ""

            if not code_to_run:
                # Bare code fallback
                if "result_df" in raw_planning_response or "df." in raw_planning_response:
                    code_to_run = raw_planning_response.strip()

            if code_to_run:
                if status_callback:
                    status_callback("exec", "🐍 Pandas / Python komutu güvenli ortamda çalıştırılıyor...")

                exec_res, py_err, retries = self._execute_python_with_self_healing(
                    code=code_to_run,
                    datasets=datasets,
                    active_key=active_key,
                    step=step,
                    status_callback=status_callback,
                )
                step.executed_code = code_to_run
                step.total_sql_retries = retries

                if exec_res:
                    step.result_df = exec_res.result_df
                    step.figures = exec_res.figures
                    step.stdout = exec_res.stdout
                if py_err:
                    step.error = f"Python Hatası: {py_err}"

        else:
            # SQL Modu: SQL sorgusunu çıkar ve SQLite üzerinde çalıştır
            planning_thought, tool_calls = ToolParser.parse_response(raw_planning_response)
            sql_query = ""
            for tc in tool_calls:
                if tc.name == "execute_sql":
                    sql_query = tc.get_arg("query", "sql", default="")
                    break

            if sql_query:
                if status_callback:
                    status_callback("sql_exec", "🔍 SQL SQLite üzerinde çalıştırılıyor...")

                df, sql_err, retries = self._execute_sql_with_self_healing(
                    sql_query=sql_query,
                    user_message=user_message,
                    step=step,
                    status_callback=status_callback,
                )
                step.sql_query = sql_query
                step.executed_code = sql_query
                step.result_df = df
                step.total_sql_retries = retries

                if sql_err:
                    step.error = f"SQL Hatası: {sql_err}"

        # ── 6. Aşama 2: Doğrulanmış Sentez (Grounded Synthesis) ──
        if step.result_df is not None and not step.result_df.empty:
            if status_callback:
                status_callback("synthesis", "✍️ Gerçek veriler inceleniyor ve doğrulanmış yönetici raporu yazılıyor...")

            grounded_report = self._generate_grounded_synthesis(
                user_message=user_message,
                sql_query=step.executed_code or "",
                result_df=step.result_df,
                intent=current_intent,
            )
            step.explanation = grounded_report

        elif step.result_df is not None and step.result_df.empty:
            step.explanation = "🔍 **Sonuç:** Yapılan analizde belirtilen kriterlere uygun herhangi bir kayıt bulunamadı."
        elif not step.error:
            step.explanation = raw_planning_response

        return step, step.explanation

    # ─────────────────────────────────────────────────────────
    # Closed-Loop Self-Healing (Python / Pandas Hata Düzeltme)
    # ─────────────────────────────────────────────────────────
    def _execute_python_with_self_healing(
        self,
        code: str,
        datasets: dict,
        active_key: str,
        step: AgentStepResult,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[Optional[ExecutionResult], Optional[str], int]:
        """Python/Pandas kodunu çalıştırır; hata durumunda LLM ile otonom düzeltir."""
        from core.prompts import build_dataframe_context
        current_code = code
        retries = 0
        active_entry = datasets.get(active_key, {})
        active_df = active_entry.get("df", pd.DataFrame())
        file_name = active_entry.get("metadata", {}).get("dosya_adi", active_key or "data.csv")

        while retries <= self.MAX_SQL_RETRIES:
            exec_res = self.executor.execute(current_code, datasets, active_key)
            if exec_res.success and (exec_res.result_df is not None or exec_res.figures or exec_res.stdout):
                return exec_res, None, retries

            retries += 1
            error_msg = exec_res.error or "result_df veya çıktı üretilmedi"
            logger.warning(f"Python/Pandas Hatası (Deneme {retries}/{self.MAX_SQL_RETRIES}): {error_msg}")

            if retries > self.MAX_SQL_RETRIES:
                return exec_res if exec_res.success else None, error_msg, retries

            if status_callback:
                status_callback(
                    "self_healing",
                    f"🩹 Python/Pandas hatası tespit edildi, otonom düzeltiliyor... ({retries}/{self.MAX_SQL_RETRIES})",
                )

            step.healing_notes.append(
                f"Düzeltme #{retries}: '{error_msg}' hatası üzerine Pandas kodu yeniden yazıldı."
            )

            df_context = build_dataframe_context(active_df, file_name)
            fix_prompt = (
                f"Aşağıdaki Python/Pandas kodu çalıştırılırken hata verdi:\n\n"
                f"❌ HATA: {error_msg}\n\n"
                f"⚠️ HATALI KOD:\n```python\n{current_code}\n```\n\n"
                f"📊 DATAFRAME GERÇEK KOLONLARI:\n{df_context}\n\n"
                f"Lütfen kodu tablodaki gerçek kolon adlarını kullanarak düzelt. "
                f"Sonucu MUTLAKA `result_df = ...` değişkenine ata.\n"
                f"SADECE düzeltilmiş ```python ... ``` bloğunu ver."
            )

            fix_messages = [
                {"role": "system", "content": "Sen hata düzelten bir Python/Pandas uzmanısın. Yalnızca düzeltilmiş ```python ... ``` bloğu ver."},
                {"role": "user", "content": fix_prompt},
            ]
            fix_response = self.llm.chat(fix_messages)
            py_matches = re.findall(r"```(?:python|py)?\s*([\s\S]*?)```", fix_response, re.IGNORECASE)
            if py_matches:
                current_code = py_matches[0].strip()
                step.executed_code = current_code
            else:
                return exec_res, error_msg, retries

        return None, "Maksimum düzeltme denemesine ulaşıldı.", retries

    # ─────────────────────────────────────────────────────────
    # Aşama 2: Doğrulanmış Sentez Motoru (Grounded Synthesis)
    # ─────────────────────────────────────────────────────────
    def _generate_grounded_synthesis(
        self,
        user_message: str,
        sql_query: str,
        result_df: pd.DataFrame,
        intent: AnalyticsIntent,
    ) -> str:
        """
        Veritabanından çekilen gerçek result_df tablosunu LLM'e vererek
        %100 matematiksel olarak doğru, halüsinasyonsuz yönetici raporu üretir.
        """
        # DataFrame'i düzenli metin / markdown formatına çevir (maksimum ilk 30 satır)
        preview_df = result_df.head(30)
        table_str = preview_df.to_string(index=False)

        if intent == AnalyticsIntent.TREND:
            template = TREND_SYNTHESIS_PROMPT
        elif intent == AnalyticsIntent.ANOMALY:
            template = ANOMALY_SYNTHESIS_PROMPT
        elif intent == AnalyticsIntent.EXPLORATION:
            template = EXPLORATION_SYNTHESIS_PROMPT
        else:
            template = GROUNDED_SYNTHESIS_PROMPT

        synthesis_prompt = template.format(
            data_table_str=table_str,
            row_count=len(result_df),
            sql_query=sql_query,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Sen %100 gerçek veritabanı sonuçlarına sadık kalarak rapor yazan Kıdemli Baş Veri Bilimcisisin. "
                    "Yalnızca verilen tablodaki gerçek sayıları, isimleri, adetleri ve oranları kullan. "
                    "Asla tabloda olmayan sayı uydurma."
                ),
            },
            {
                "role": "user",
                "content": f"Kullanıcı Sorusu: {user_message}\n\n{synthesis_prompt}",
            },
        ]

        try:
            report = self.llm.chat(messages)
            return report.strip()
        except Exception as e:
            logger.error(f"Grounded synthesis LLM call failed: {e}")
            # Fallback olarak doğrudan tabloyu ve özet metni döndür
            return f"### 📊 Temel Bulgular\n\nSorgu sonucunda {len(result_df)} kayıt listelenmiştir:\n\n```\n{table_str}\n```"

    # ─────────────────────────────────────────────────────────
    # Closed-Loop Self-Healing (SQL Hata Düzeltme)
    # ─────────────────────────────────────────────────────────
    def _execute_sql_with_self_healing(
        self,
        sql_query: str,
        user_message: str,
        step: AgentStepResult,
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str], int]:
        """SQL sorgusunu çalıştırır; hata durumunda LLM ile otonom olarak düzeltir."""
        current_sql = sql_query
        retries = 0

        while retries <= self.MAX_SQL_RETRIES:
            try:
                df = self.db_manager.execute_query(current_sql)
                return df, None, retries

            except (sqlite3.Error, ValueError, Exception) as e:
                retries += 1
                error_msg = str(e)
                logger.warning(f"SQL Hatası (Deneme {retries}/{self.MAX_SQL_RETRIES}): {error_msg}")

                if retries > self.MAX_SQL_RETRIES:
                    return None, error_msg, retries

                if status_callback:
                    status_callback(
                        "self_healing",
                        f"🩹 SQL hatası tespit edildi, otonom düzeltiliyor... ({retries}/{self.MAX_SQL_RETRIES})",
                    )

                step.healing_notes.append(
                    f"Düzeltme #{retries}: '{error_msg}' hatası üzerine SQL yeniden planlandı."
                )

                table_cols_hints = []
                for t in self.db_manager.get_table_names():
                    cols = self.db_manager.get_table_columns(t)
                    table_cols_hints.append(f"- Tablo `{t}` içindeki GERÇEK kolonlar: {', '.join(f'`{c}`' for c in cols)}")

                cols_reminder = "\n".join(table_cols_hints)

                custom_advice = ""
                if "no such column" in error_msg.lower():
                    custom_advice = (
                        "\n⚠️ KOLON BULUNAMADI UYARISI:\n"
                        "Kullandığın kolon adı veritabanında mevcut değil! "
                        "Lütfen yukarıda listelenen GERÇEK kolon adlarından birini seç. "
                        "Eğer kullanıcı 'marka' sorduysa ama tabloda 'Manufacturer'/'Marka' yoksa (çünkü veri seti zaten tek bir markaya aittir, örn: BMW), "
                        "'Model', 'Series', 'Segment' veya 'Region' gibi mevcut bir kategorik kolona göre gruplama yaparak soruyu yanıtla."
                    )

                fix_prompt = (
                    f"Önceki SQL sorgusu şu hatayı verdi:\n"
                    f"❌ HATA: {error_msg}\n\n"
                    f"⚠️ HATALI SQL:\n```sql\n{current_sql}\n```\n\n"
                    f"📊 VERİTABANINDA MEVCUT OLAN GERÇEK KOLONLAR:\n{cols_reminder}\n"
                    f"{custom_advice}\n\n"
                    f"Lütfen bu hatayı analiz et ve YALNIZCA düzeltilmiş çalışan ```sql ... ``` kodunu ver."
                )

                fix_messages = [
                    {"role": "system", "content": "Sen hata düzelten bir SQL uzmanısın. Yalnızca düzeltilmiş ```sql ... ``` bloğu ver."},
                    {"role": "user", "content": fix_prompt},
                ]
                fix_response = self.llm.chat(fix_messages)
                _, new_tools = ToolParser.parse_response(fix_response)

                corrected = False
                for t in new_tools:
                    if t.name == "execute_sql" and t.get_arg("query", "sql"):
                        current_sql = t.get_arg("query", "sql")
                        step.sql_query = current_sql
                        corrected = True
                        break

                if not corrected:
                    return None, error_msg, retries

        return None, "Maksimum düzeltme denemesine ulaşıldı.", retries

    # ─────────────────────────────────────────────────────────
    # Closed-Loop Self-Healing (Görselleştirme Hata Düzeltme)
    # ─────────────────────────────────────────────────────────
    def _execute_plot_with_self_healing(
        self,
        code: str,
        result_df: pd.DataFrame,
        step: AgentStepResult,
    ) -> Tuple[Optional[go.Figure], Optional[str]]:
        """Görselleştirme kodunu çalıştırır; hata olursa LLM ile otonom düzeltir."""
        current_code = code

        for retry in range(self.MAX_PLOT_RETRIES + 1):
            exec_res: ExecutionResult = self.executor.execute_plot(current_code, result_df)
            if exec_res.success and exec_res.figures:
                return exec_res.figures[0], None

            err = exec_res.error
            if retry < self.MAX_PLOT_RETRIES and err:
                step.healing_notes.append(f"Görselleştirme düzeltmesi #{retry + 1}: {err}")
                fix_prompt = (
                    f"Aşağıdaki Plotly görselleştirme kodu hata verdi:\n"
                    f"Hata: {err}\n\n"
                    f"result_df kolonları: {list(result_df.columns)}\n"
                    f"Hatalı kod:\n```python\n{current_code}\n```\n"
                    f"Lütfen SADECE düzeltilmiş ```python ... ``` kodunu ver."
                )
                fix_resp = self.llm.chat([
                    {"role": "system", "content": "Yalnızca düzeltilmiş ```python ... ``` bloğu döndür."},
                    {"role": "user", "content": fix_prompt},
                ])
                _, tools = ToolParser.parse_response(fix_resp)
                found = False
                for t in tools:
                    if t.name == "generate_python_plot" and t.get_arg("code", "python_code"):
                        current_code = t.get_arg("code", "python_code")
                        found = True
                        break
                if not found:
                    break

        return None, err

    # ─────────────────────────────────────────────────────────
    # Otomatik Fallback Plot Oluşturucu
    # ─────────────────────────────────────────────────────────
    def _auto_generate_fallback_plot(
        self,
        df: pd.DataFrame,
        user_query: str,
    ) -> Optional[go.Figure]:
        """Eğer LLM grafik kodu üretmediyse ama kullanıcı açıkça istediyse akıllı fallback üret."""
        try:
            import plotly.express as px
            cols = df.columns.tolist()
            if len(cols) >= 2:
                x_col = cols[0]
                y_col = cols[1]
                fig = px.bar(
                    df.head(15),
                    x=x_col,
                    y=y_col,
                    title="Analiz Sonucu",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                return fig
        except Exception as e:
            logger.debug(f"Fallback plot generation failed: {e}")
        return None

    def is_llm_available(self) -> bool:
        """LLM sunucusunun erişilebilirliğini kontrol et."""
        return self.llm.is_available()
