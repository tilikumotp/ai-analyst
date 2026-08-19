"""
SQL ReAct Agent — Text-to-SQL, Semantic Layer, Dinamik Intent Router & Self-Healing Orkestratörü.

Akış:
1. Kullanıcı sorusunu al ve IntentRouter ile sınıflandır (Trend, Anomali, Keşif veya Genel Analitik).
2. Semantic Retrieval Engine ile ilgili kurumsal iş kurallarını (Business Glossary) çek.
3. Pre-Flight Guardrail ile çelişki denetimi yap (Sıfır maliyetle erken çıkış).
4. İlgili Persona Prompt Şablonu, Şema ve İş Kuralları ile prompt hazırla.
5. LLM'den ReAct planı, SQL sorgusu ve opsiyonel Plotly kodu al.
6. Deterministik SQLite motorunda SQL'i çalıştır (execute_sql).
7. Hata durumunda Closed-Loop Self-Healing döngüsünü işlet.
8. Dönen result_df üzerinde görselleştirme kodunu çalıştır (generate_python_plot).
9. Yönetici Özeti, Tablo, Görselleştirme ve Data Governance detaylarını kullanıcıya sun.
"""

import logging
import sqlite3
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
    ANOMALY_RISK_PROMPT,
    AUTONOMOUS_EXPLORER_PROMPT,
    SQL_ERROR_FIX_PROMPT,
    SQL_REACT_SYSTEM_PROMPT,
    TREND_ANALYST_PROMPT,
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
    """Text-to-SQL, Semantik Katman ve Dinamik Intent Router tabanlı akıllı veri analisti ajanı."""

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
    # Mesaj Listesi Oluşturma (Persona + Şema + Semantik Kurallar)
    # ─────────────────────────────────────────────────────────
    def build_messages(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        relevant_metrics: Optional[List[BusinessMetric]] = None,
    ) -> Tuple[List[Dict[str, str]], str]:
        """LLM için niyet yönlendirmeli sistem yönergesi + semantik kurallar + şema + geçmiş derle."""
        schema_context = self.db_manager.get_schema_context()

        # Semantik kurallar metni oluştur
        if relevant_metrics:
            rules_parts = []
            for m in relevant_metrics:
                rules_parts.append(
                    f"📌 **{m.canonical_name}** (v{m.version} • Sahip: {m.owner})\n"
                    f"• Resmi Tanım: {m.business_definition}\n"
                    f"• Teknik Formül / SQL: `{m.sql_formula}`\n"
                    f"• Zorunlu Filtreler (Mandatory): `{m.mandatory_filters or 'Yok'}`\n"
                    f"• Eş Anlamlılar: {', '.join(m.aliases)}"
                )
            semantic_context = "\n\n".join(rules_parts)
        else:
            semantic_context = "Bu soru için özel bir iş kuralı eşleşmedi. Genel SQL ve veri analitiği mantığını kullan."

        # Dinamik Niyet Yönlendirme (Intent Routing)
        intent, persona_label = IntentRouter.route(user_message)
        if intent == AnalyticsIntent.TREND:
            template = TREND_ANALYST_PROMPT
        elif intent == AnalyticsIntent.ANOMALY:
            template = ANOMALY_RISK_PROMPT
        elif intent == AnalyticsIntent.EXPLORATION:
            template = AUTONOMOUS_EXPLORER_PROMPT
        else:
            template = SQL_REACT_SYSTEM_PROMPT

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
        return messages, persona_label

    # ─────────────────────────────────────────────────────────
    # ReAct Döngüsü
    # ─────────────────────────────────────────────────────────
    def execute_react_cycle(
        self,
        user_message: str,
        chat_history: List[Dict[str, str]],
        status_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Tuple[AgentStepResult, str]:
        """
        ReAct (Intent Route -> Semantic Retrieve -> Pre-Flight Guardrail -> Reason -> Act -> Observe) döngüsü.
        """
        step = AgentStepResult()

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

        # ── 3. Pre-Flight Guardrail & Çatışma Denetimi (Zero-Cost Early Return) ──
        is_blocked, block_msg = self.guardrail_engine.pre_flight_check(user_message, relevant_metrics)
        if is_blocked and block_msg:
            logger.info(f"Pre-Flight Guardrail Tetiklendi: {block_msg}")
            step.explanation = block_msg
            step.guardrail_warnings = [block_msg]
            if status_callback:
                status_callback("guardrail", "🛡️ Pre-Flight Guardrail: Kurumsal kural çatışması engellendi.")
            return step, block_msg

        if status_callback:
            if relevant_metrics:
                rule_names = ", ".join(f"'{m.canonical_name}'" for m in relevant_metrics)
                status_callback("reasoning", f"🧠 {rule_names} kuralı uygulandı, {persona_label} ile SQL planlanıyor...")
            else:
                status_callback("reasoning", f"🧠 {persona_label} soruyu analiz ediyor ve SQL planlıyor...")

        # ── 4. Prompt Hazırla ve LLM'e Gönder (Reason + Plan) ──
        messages, _ = self.build_messages(user_message, chat_history, relevant_metrics)
        raw_llm_response = self.llm.chat(messages)

        # ── 5. Tool Çağrılarını Ayrıştır (Parse Tools) ──
        explanation, tool_calls = ToolParser.parse_response(raw_llm_response)
        step.explanation = explanation

        # ── 6. Araçları Yürüt (Act) ──
        for tool_call in tool_calls:
            if tool_call.name == "execute_sql":
                sql_query = tool_call.arguments.get("query", "")
                if sql_query:
                    if status_callback:
                        status_callback("sql_exec", f"🔍 SQL SQLite üzerinde çalıştırılıyor...")

                    df, sql_err, retries = self._execute_sql_with_self_healing(
                        sql_query=sql_query,
                        user_message=user_message,
                        step=step,
                        status_callback=status_callback,
                    )
                    step.sql_query = sql_query
                    step.result_df = df
                    step.total_sql_retries = retries

                    if sql_err:
                        step.error = f"SQL Hatası: {sql_err}"
                        logger.error(f"SQL execution failed: {sql_err}")

            elif tool_call.name == "generate_python_plot":
                plot_code = tool_call.arguments.get("code", "")
                if plot_code:
                    step.plot_code = plot_code
                    if step.result_df is not None and not step.result_df.empty:
                        if status_callback:
                            status_callback("plotting", "📊 Plotly görselleştirmesi üretiliyor...")

                        fig, plot_err = self._execute_plot_with_self_healing(
                            code=plot_code,
                            result_df=step.result_df,
                            step=step,
                        )
                        if fig:
                            step.figures.append(fig)
                        if plot_err and not fig:
                            logger.warning(f"Plot generation failed: {plot_err}")

        # Eğer tool parser ile bulunamadıysa ama result_df varsa ve kullanıcı grafik istemişse
        if step.result_df is not None and not step.result_df.empty and not step.has_figure:
            q_lower = user_message.lower()
            if any(w in q_lower for w in ("grafik", "çiz", "görselleştir", "plot", "bar", "dağılım", "trend")):
                fallback_fig = self._auto_generate_fallback_plot(step.result_df, user_message)
                if fallback_fig:
                    step.figures.append(fallback_fig)

        return step, raw_llm_response

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

                fix_prompt = SQL_ERROR_FIX_PROMPT.format(
                    error_message=error_msg,
                    code_type="sql",
                    failed_code=current_sql,
                    schema_context=self.db_manager.get_schema_context(),
                )

                fix_messages = [
                    {"role": "system", "content": "Sen hata düzelten bir SQL uzmanısın. Yalnızca düzeltilmiş ```sql ... ``` bloğu ver."},
                    {"role": "user", "content": fix_prompt},
                ]
                fix_response = self.llm.chat(fix_messages)
                _, new_tools = ToolParser.parse_response(fix_response)

                corrected = False
                for t in new_tools:
                    if t.name == "execute_sql" and t.arguments.get("query"):
                        current_sql = t.arguments["query"]
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
            fig, err = self.executor.execute_plot(current_code, result_df)
            if fig is not None and not err:
                return fig, None

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
                    if t.name == "generate_python_plot" and t.arguments.get("code"):
                        current_code = t.arguments["code"]
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
                # 1. kolon kategorik / metin, 2. kolon sayısal
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
