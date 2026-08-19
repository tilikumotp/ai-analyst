"""
🧠 AI Data Analyst — Text-to-SQL, Semantic Layer & ReAct Tool Calling Platform

Ana Streamlit uygulaması.
Çalıştırma: streamlit run app.py
"""
import os
import re
import uuid
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
import pandas as pd

from core.sql_agent import SQLReActAgent, AgentStepResult
from core.knowledge_base import KnowledgeBaseManager, BusinessMetric
from core.prompts import WELCOME_MESSAGE, QUICK_PROMPTS
from core.history import ChatHistory
from ingestion.csv_loader import CSVLoader
from ingestion.db_loader import DatabaseManager
from ingestion.data_profiler import DataHealthProfiler


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sayfa Konfigürasyonu
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="AI Data Analyst (Semantic Layer)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Premium CSS Teması
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d29 0%, #131620 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.12);
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #a5b4fc;
    text-transform: uppercase;
    margin-top: 1.2rem;
}

/* ── Header Gradient ── */
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, #1a1d29 0%, #0f1117 100%);
    border-bottom: 1px solid rgba(99, 102, 241, 0.12);
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    margin-bottom: 8px;
    backdrop-filter: blur(8px);
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.25);
    background: rgba(99, 102, 241, 0.08);
    color: #c7d2fe;
    font-weight: 500;
    font-size: 0.82rem;
    padding: 8px 16px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    text-align: left;
}

.stButton > button:hover {
    background: rgba(99, 102, 241, 0.18);
    border-color: #6366f1;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.15);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    font-weight: 600;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}

[data-testid="stFileUploader"] > div > div {
    border: 2px dashed rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    transition: border-color 0.3s ease;
}

[data-testid="stFileUploader"] > div > div:hover {
    border-color: #6366f1;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: rgba(99, 102, 241, 0.06);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 10px;
    padding: 12px 16px;
}

[data-testid="stMetricValue"] {
    font-size: 1.4rem;
    font-weight: 700;
    color: #a5b4fc;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(99, 102, 241, 0.08);
}

/* ── Code Blocks ── */
.stCodeBlock {
    border-radius: 10px;
    border: 1px solid rgba(99, 102, 241, 0.08);
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    border-radius: 12px;
}

[data-testid="stChatInput"] textarea {
    border-radius: 12px;
    border: 1px solid rgba(99, 102, 241, 0.2);
    font-family: 'Inter', sans-serif;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15);
}

/* ── Governance Card ── */
.governance-card {
    background: rgba(16, 185, 129, 0.06);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
}

.governance-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #34d399;
    margin-bottom: 6px;
}

/* ── Welcome Hero ── */
.welcome-hero {
    text-align: center;
    padding: 48px 24px 32px;
    animation: fadeIn 0.8s ease-out;
}

.welcome-hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}

.welcome-hero p {
    color: #9ca3af;
    font-size: 1.05rem;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}

.feature-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.04) 100%);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 14px;
    padding: 24px 18px;
    text-align: center;
    transition: all 0.3s ease;
    min-height: 165px;
}

.feature-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.1);
}

.feature-icon {
    font-size: 2.2rem;
    margin-bottom: 12px;
}

.feature-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e4e6f0;
    margin-bottom: 6px;
}

.feature-desc {
    color: #9ca3af;
    font-size: 0.8rem;
    line-height: 1.5;
}

/* ── Sidebar Logo ── */
.sidebar-brand {
    text-align: center;
    padding: 16px 0 8px;
}

.sidebar-brand h1 {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

.sidebar-brand p {
    color: #6b7280;
    font-size: 0.75rem;
    margin: 2px 0 0;
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}

.status-online {
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.status-offline {
    background: rgba(239, 68, 68, 0.12);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

/* ── Divider ── */
.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(99, 102, 241, 0.1);
    margin: 16px 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Session State Başlatma
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

if "kb_manager" not in st.session_state:
    st.session_state.kb_manager = KnowledgeBaseManager()

if "agent" not in st.session_state:
    st.session_state.agent = SQLReActAgent(
        db_manager=st.session_state.db_manager,
        kb_manager=st.session_state.kb_manager,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "datasets" not in st.session_state:
    st.session_state.datasets = {}          # {"key": {"df": DataFrame, "metadata": dict, "sql_table": str}}

if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = None  # Aktif dataset anahtarı

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "history" not in st.session_state:
    st.session_state.history = ChatHistory()

if "session_id" not in st.session_state:
    st.session_state.session_id = ChatHistory.generate_id()

MAX_DATASETS = 5  # Maksimum yüklenebilir dosya sayısı


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Yardımcı Fonksiyonlar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _save_current_session():
    """Mevcut sohbeti geçmişe kaydet."""
    if not st.session_state.messages:
        return

    file_names = [
        entry["metadata"].get("dosya_adi")
        for entry in st.session_state.datasets.values()
        if entry.get("metadata")
    ]

    st.session_state.history.save_session(
        session_id=st.session_state.session_id,
        messages=st.session_state.messages,
        file_names=file_names if file_names else None,
    )


def _get_active_df() -> Optional[pd.DataFrame]:
    key = st.session_state.active_dataset
    if key and key in st.session_state.datasets:
        return st.session_state.datasets[key]["df"]
    return None


def _get_active_meta() -> Optional[dict]:
    key = st.session_state.active_dataset
    if key and key in st.session_state.datasets:
        return st.session_state.datasets[key].get("metadata")
    return None


def _has_datasets() -> bool:
    return len(st.session_state.datasets) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:

    # ── Branding ──
    st.markdown(
        """
    <div class="sidebar-brand">
        <h1>🧠 AI Data Analyst</h1>
        <p>Text-to-SQL • Semantic Layer • ReAct</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Dosya Yükleme ──
    st.markdown("### 📂 Veri Setleri (SQLite)")

    can_upload = len(st.session_state.datasets) < MAX_DATASETS

    uploaded_file = st.file_uploader(
        "CSV veya Excel dosyanızı yükleyin" if can_upload else f"Maksimum {MAX_DATASETS} dosya yüklendi",
        type=["csv", "xlsx", "xls"],
        help="CSV ve Excel dosyaları otomatik optimize SQLite tablosuna dönüştürülür.",
        disabled=not can_upload,
    )

    if uploaded_file:
        file_name_lower = uploaded_file.name.lower()
        is_excel = file_name_lower.endswith((".xlsx", ".xls"))

        selected_sheet = None
        if is_excel:
            sheet_names = CSVLoader.get_sheet_names(uploaded_file)
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "📑 Sheet Seçin",
                    sheet_names,
                    key="sheet_selector",
                )
            uploaded_file.seek(0)

        dataset_key = uploaded_file.name
        already_loaded = dataset_key in st.session_state.datasets

        sheet_changed = (
            is_excel
            and already_loaded
            and st.session_state.datasets[dataset_key].get("metadata", {}).get("sheet") != selected_sheet
            and selected_sheet is not None
        )

        if not already_loaded or sheet_changed:
            with st.spinner("📊 Dosya işleniyor ve SQLite'a aktarılıyor..."):
                try:
                    df, metadata = CSVLoader.load(uploaded_file, selected_sheet)
                    sql_table = st.session_state.db_manager.load_dataframe(df, uploaded_file.name)
                    metadata["sql_table"] = sql_table

                    st.session_state.datasets[dataset_key] = {
                        "df": df,
                        "metadata": metadata,
                        "sql_table": sql_table,
                    }
                    st.session_state.active_dataset = dataset_key
                    st.toast(f"✅ `{uploaded_file.name}` yüklendi (Tablo: `{sql_table}`)", icon="📊")
                except Exception as e:
                    st.error(f"❌ Dosya yükleme hatası: {e}")

    # ── Yüklü Veri Setleri Listesi ──
    if _has_datasets():
        st.markdown("---")
        st.caption(f"📊 **{len(st.session_state.datasets)}** tablo SQLite üzerinde hazır")

        for ds_key, ds_entry in list(st.session_state.datasets.items()):
            meta = ds_entry.get("metadata", {})
            ds_df = ds_entry["df"]
            sql_tbl = ds_entry.get("sql_table", "")
            is_active = (ds_key == st.session_state.active_dataset)
            file_name = meta.get("dosya_adi", ds_key)

            pin = "📌 " if is_active else ""
            row_info = f"`{sql_tbl}` • {len(ds_df):,} satır"

            col_info, col_act, col_del = st.columns([4, 1, 1])

            with col_info:
                if is_active:
                    st.markdown(f"**{pin}{file_name}**")
                else:
                    st.caption(f"📄 {file_name}")
                st.caption(row_info)

            with col_act:
                if not is_active:
                    if st.button("📌", key=f"pin_{ds_key}", help="Aktif yap"):
                        st.session_state.active_dataset = ds_key
                        st.rerun()

            with col_del:
                if st.button("🗑", key=f"rds_{ds_key}", help="Kaldır"):
                    st.session_state.db_manager.remove_table(ds_key)
                    del st.session_state.datasets[ds_key]
                    if st.session_state.active_dataset == ds_key:
                        if st.session_state.datasets:
                            st.session_state.active_dataset = next(iter(st.session_state.datasets))
                        else:
                            st.session_state.active_dataset = None
                    st.rerun()

        # ── Aktif Tablo Özeti ──
        active_df = _get_active_df()
        active_meta = _get_active_meta()

        if active_df is not None and active_meta:
            meta = active_meta
            st.markdown("---")
            file_type_emoji = "📄" if meta.get("dosya_tipi") == "CSV" else "📗"
            st.success(f"{file_type_emoji} **Tablo: `{meta.get('sql_table', meta['dosya_adi'])}`**")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Satır", f"{meta['satir_sayisi']:,}")
            with col2:
                st.metric("Kolon", meta["kolon_sayisi"])

            with st.expander("🔍 Tablo Şeması & Kolonlar", expanded=False):
                for col_name in active_df.columns:
                    col_data = active_df[col_name]
                    dtype = str(col_data.dtype)
                    nulls = int(col_data.isnull().sum())
                    uniques = int(col_data.nunique())
                    null_badge = f"  ⚠️ {nulls} eksik" if nulls > 0 else ""
                    st.markdown(f"• **{col_name}** `{dtype}` ({uniques} tekil){null_badge}")

            csv_data = active_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 Orijinal Veriyi İndir (CSV)",
                data=csv_data,
                file_name=f"{meta['dosya_adi']}_export.csv",
                mime="text/csv",
                key="download_csv",
            )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Semantik Katman & İş Kuralları (Business Glossary) ──
    st.markdown("### 📚 Semantik Katman (İş Kuralları)")
    metrics_list = st.session_state.kb_manager.list_all_metrics()
    active_count = len([m for m in metrics_list if m.is_active])
    st.caption(f"🛡️ **{active_count}/{len(metrics_list)}** kurumsal iş kuralı aktif")

    with st.expander("📖 Tanımlı Kuralları İncele", expanded=False):
        for m in metrics_list:
            status_icon = "🟢" if m.is_active else "⚪ (Pasif)"
            st.markdown(f"**📌 {m.canonical_name}** `v{m.version}` {status_icon}")
            st.caption(f"🏢 Sahip: *{m.owner}*")
            st.caption(f"📖 *{m.business_definition}*")
            st.code(f"Formül: {m.sql_formula}\nFiltre: {m.mandatory_filters or 'Yok'}", language="sql")
            st.markdown("---")

    with st.expander("➕ Yeni İş Kuralı Ekle", expanded=False):
        with st.form("add_metric_form"):
            new_name = st.text_input("Resmi Metrik Adı", placeholder="örn: Net Kar Marjı")
            new_owner = st.text_input("Sorumlu Departman", placeholder="örn: Finans Departmanı", value="Veri Yönetişimi")
            new_version = st.text_input("Versiyon", placeholder="1.0", value="1.0")
            new_aliases = st.text_input("Eş Anlamlılar (Virgülle ayırın)", placeholder="kar marji, net margin, karlilik")
            new_def = st.text_area("İş Tanımı", placeholder="Net karın toplam ciroya bölünmesiyle hesaplanır.")
            new_formula = st.text_input("SQL Formülü", placeholder="SUM(net_kar) / SUM(ciro) * 100")
            new_filter = st.text_input("Zorunlu Filtre (Opsiyonel)", placeholder="siparis_durumu = 'ONAYLANDI'")

            if st.form_submit_button("💾 Kuralı Kaydet"):
                if new_name and new_formula:
                    metric_obj = BusinessMetric(
                        canonical_name=new_name.strip(),
                        aliases=[a.strip() for a in new_aliases.split(",") if a.strip()],
                        business_definition=new_def.strip(),
                        sql_formula=new_formula.strip(),
                        mandatory_filters=new_filter.strip(),
                        version=new_version.strip() or "1.0",
                        owner=new_owner.strip() or "Veri Yönetişimi",
                        is_active=True,
                    )
                    st.session_state.kb_manager.add_or_update_metric(metric_obj)
                    st.toast(f"✅ '{new_name}' (v{metric_obj.version}) kaydedildi!", icon="🛡️")
                    st.rerun()
                else:
                    st.error("Metrik Adı ve SQL Formülü zorunludur.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Yeni Sohbet ──
    if st.button("🔄 Yeni Sohbet Başlat", width="stretch"):
        if st.session_state.messages:
            _save_current_session()
        st.session_state.messages = []
        st.session_state.session_id = ChatHistory.generate_id()
        st.rerun()

    # ── Tüm Veri Setlerini Sıfırla ──
    if _has_datasets():
        if st.button("🗑️ Tüm Tabloları Temizle", width="stretch"):
            if st.session_state.messages:
                _save_current_session()
            st.session_state.db_manager.clear_all()
            st.session_state.datasets = {}
            st.session_state.active_dataset = None
            st.session_state.messages = []
            st.session_state.session_id = ChatHistory.generate_id()
            st.rerun()

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Sohbet Geçmişi ──
    st.markdown("### 📜 Geçmiş Oturumlar")
    sessions = st.session_state.history.list_sessions(limit=10)

    if sessions:
        for sess in sessions:
            title = sess["title"]
            msg_count = sess.get("message_count", 0)
            file_names = sess.get("file_names", [])
            file_badge = f" 📄 {', '.join(file_names)}" if file_names else ""

            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(f"💬 {title}", key=f"load_{sess['id']}", help=f"{msg_count} mesaj{file_badge}"):
                    if st.session_state.messages:
                        _save_current_session()
                    loaded = st.session_state.history.load_session(sess["id"])
                    if loaded:
                        st.session_state.messages = [
                            {"role": m["role"], "content": m["content"]}
                            for m in loaded.get("messages", [])
                        ]
                        st.session_state.session_id = sess["id"]
                        st.session_state.loaded_file_names = loaded.get("file_names", [])
                        st.rerun()

            with col_del:
                if st.button("🗑", key=f"del_{sess['id']}", help="Sil"):
                    st.session_state.history.delete_session(sess["id"])
                    st.rerun()
    else:
        st.caption("Henüz kayıtlı oturum yok.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Sunucu Durumu ──
    st.markdown("### ⚙️ Sistem Durumu")
    try:
        llm_ok = st.session_state.agent.is_llm_available()
        model_name = st.session_state.agent.llm.get_model_info() if llm_ok else ""
    except Exception:
        llm_ok = False
        model_name = ""

    if llm_ok:
        st.markdown('<div class="status-badge status-online">● LM Studio Aktif</div>', unsafe_allow_html=True)
        if model_name:
            st.caption(f"🤖 Model: `{model_name}`")
    else:
        st.markdown('<div class="status-badge status-offline">● LM Studio Bağlantısız</div>', unsafe_allow_html=True)
        st.caption("LM Studio'da **Local Server**'ı (`http://127.0.0.1:1234`) başlatın ve bir model yükleyin.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANA İÇERİK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if not _has_datasets():

    if st.session_state.messages:
        loaded_files = getattr(st.session_state, "loaded_file_names", []) or []
        files_str = ", ".join(f"`{f}`" for f in loaded_files) if loaded_files else "Yok"
        st.info(f"📜 **Geçmiş oturum yüklendi** — Tablolar: {files_str}\n\nYeni soru sormak için dosya yükleyin.", icon="📂")

        for idx, msg in enumerate(st.session_state.messages):
            avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg.get("content", ""))
                for fig in msg.get("figures", []):
                    st.plotly_chart(fig, width="stretch")
                if msg.get("sql_query"):
                    with st.expander("🔍 SQL Sorgusu ve Sonuç", expanded=False):
                        st.code(msg["sql_query"], language="sql")
                        if msg.get("result_df") is not None and not msg["result_df"].empty:
                            st.dataframe(msg["result_df"], width="stretch")

    else:
        # ━━━ Karşılama Ekranı ━━━
        if os.path.exists("ai-analyst.png"):
            st.image("ai-analyst.png", width="stretch")

        st.markdown(
            """
        <div class="welcome-hero">
            <h1>AI Data Analyst</h1>
            <p>Text-to-SQL, Semantik Katman (Semantic Layer) ve ReAct Tool Calling ile kurumsal veri analitiği.<br/>
            Şirket iş kurallarına sadık, deterministik ve denetlenebilir analitik motoru.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        cols = st.columns(4)
        features = [
            ("🛡️", "Semantik Katman", "Resmi metrikler (Net Ciro, AOV vb.) ve zorunlu filtreler"),
            ("🏛️", "Text-to-SQL", "Deterministik SQLite sorguları ile sıfır halüsinasyon"),
            ("🛠️", "ReAct Tool Calling", "Adım adım düşünme, araç seçme ve otonom karar"),
            ("🩹", "Closed-Loop Healing", "SQL ve kod sözdizimi hatalarını otonom düzeltme"),
        ]

        for i, (icon, title, desc) in enumerate(features):
            with cols[i]:
                st.markdown(
                    f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("")
        st.markdown("")
        st.info("👈 Sol panelden bir **CSV veya Excel dosyası** yükleyerek başlayın.", icon="📂")


else:

    # ━━━ Analiz Görünümü ━━━
    current_df = _get_active_df()

    # ── Veri Önizleme ──
    with st.expander("📊 SQLite Veri & Tablo Önizleme", expanded=False):
        if len(st.session_state.datasets) > 1:
            ds_keys = list(st.session_state.datasets.keys())
            active_idx = ds_keys.index(st.session_state.active_dataset) if st.session_state.active_dataset in ds_keys else 0
            selected_ds = st.selectbox("📂 Tablo Seçin", ds_keys, index=active_idx, key="preview_dataset_selector")
            preview_df = st.session_state.datasets[selected_ds]["df"]
        else:
            preview_df = current_df

        if preview_df is not None:
            tab_table, tab_stats = st.tabs(["📄 Tablo (İlk 100)", "📈 İstatistikler"])
            with tab_table:
                st.dataframe(preview_df.head(100), width="stretch", height=320)
            with tab_stats:
                numeric_df = preview_df.select_dtypes(include=["number"])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe().round(2), width="stretch")
                else:
                    st.info("Sayısal kolon bulunamadı.")

    # ── Veri Sağlık ve Hazırlık Raporu (Data Health Profiler) ──
    if current_df is not None:
        profile = DataHealthProfiler.profile(current_df)
        with st.expander(f"🩺 Veri Sağlık & Kalite Raporu — {profile['status_badge']} ({profile['health_score']}/100)", expanded=False):
            st.markdown(f"**Değerlendirme:** {profile['status_desc']}")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Sağlık Skoru", f"{profile['health_score']} / 100")
            with c2:
                missing_total = sum(m['count'] for m in profile['missing_cols'])
                st.metric("Eksik Hücre", f"{missing_total:,}")
            with c3:
                outlier_total = sum(o['outlier_count'] for o in profile['outlier_cols'])
                st.metric("Aykırı Değer (IQR)", f"{outlier_total:,}")
            with c4:
                st.metric("Mükerrer Satır", f"{profile['duplicate_count']:,}")

            st.markdown("---")
            st.markdown("##### 📋 Tespit Edilen Bulgular:")
            for f in profile['findings']:
                st.markdown(f"• {f}")

            st.markdown("##### 💡 Veri Hazırlama & Stratejik Öneriler:")
            for r in profile['recommendations']:
                st.markdown(f"• {r}")

    # ── Hızlı Başlangıç Butonları ──
    if not st.session_state.messages:
        st.markdown("### 💡 Hızlı Başlangıç")
        st.caption("Bir öneri seçin veya aşağıdaki alana kendi sorunuzu yazın:")

        qp_cols = st.columns(3)
        for i, qp in enumerate(QUICK_PROMPTS):
            with qp_cols[i % 3]:
                if st.button(qp, key=f"quick_{i}", width="stretch"):
                    st.session_state.pending_prompt = qp
                    st.rerun()

    # ── Sohbet Geçmişini Göster ──
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg.get("content", ""))

            # Guardrail Uyarıları
            guardrail_warns = msg.get("guardrail_warnings", [])
            for warn in guardrail_warns:
                st.warning(warn)

            # Grafikleri göster
            for fig in msg.get("figures", []):
                st.plotly_chart(fig, width="stretch")

            # Teknik Detaylar ve Yönetişim
            applied_metrics = msg.get("applied_metrics", [])
            sql_query = msg.get("sql_query")
            executed_code = msg.get("executed_code") or sql_query
            code_type = msg.get("code_type", "python" if not sql_query else "sql")
            res_df = msg.get("result_df")

            if applied_metrics or executed_code or (res_df is not None and not res_df.empty):
                with st.expander("🔍 Teknik Detayları ve Çalıştırılan Kodu Göster", expanded=False):
                    if applied_metrics:
                        st.markdown("##### 🛡️ Uygulanan Şirket İş Kuralları (Data Governance)")
                        for m in applied_metrics:
                            c_name = m.get("canonical_name", "") if isinstance(m, dict) else getattr(m, "canonical_name", "")
                            b_def = m.get("business_definition", "") if isinstance(m, dict) else getattr(m, "business_definition", "")
                            s_form = m.get("sql_formula", "") if isinstance(m, dict) else getattr(m, "sql_formula", "")
                            m_filt = m.get("mandatory_filters", "") if isinstance(m, dict) else getattr(m, "mandatory_filters", "")
                            ver = m.get("version", "1.0") if isinstance(m, dict) else getattr(m, "version", "1.0")
                            owner = m.get("owner", "Veri Yönetişimi") if isinstance(m, dict) else getattr(m, "owner", "Veri Yönetişimi")

                            st.markdown(f"• **{c_name}** `v{ver}` • *{owner}*")
                            st.caption(f"**Tanım:** {b_def}")
                            st.code(f"Formül: {s_form}\nFiltre: {m_filt or 'Yok'}", language="sql")
                            st.markdown("---")

                    if executed_code:
                        if code_type == "python":
                            st.markdown("##### 🐍 Çalıştırılan Pandas / Python Kodu")
                            st.code(executed_code, language="python")
                        else:
                            st.markdown("##### 💾 Çalıştırılan SQL Sorgusu")
                            st.code(executed_code, language="sql")

                    if res_df is not None and not res_df.empty:
                        st.markdown(f"##### 📊 Sonuç Tablosu ({len(res_df):,} kayıt)")
                        st.dataframe(res_df, width="stretch")

                        csv_bytes = res_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="📥 Bu Tabloyu İndir (CSV)",
                            data=csv_bytes,
                            file_name=f"analiz_sonucu_{idx}.csv",
                            mime="text/csv",
                            key=f"dl_sql_{idx}",
                        )

            if msg.get("error"):
                st.error(msg["error"])

    # ── Kullanıcı Mesaj Girişi ──
    prompt = st.chat_input("Veri setiniz hakkında bir soru sorun (örn: 'En çok satan ilk 5 marka hangisi?')...")

    if not prompt and st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    # ── Prompt İşleme (ReAct Döngüsü) ──
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):

            if not st.session_state.agent.is_llm_available():
                error_msg = (
                    "❌ **LM Studio sunucusuna bağlanılamıyor.**\n\n"
                    "Lütfen **LM Studio** uygulamasında **Local Server** sekmesinden sunucuyu (`http://127.0.0.1:1234`) başlatın ve bir model yükleyin."
                )
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

            else:
                with st.status("🧠 ReAct & Veri Analiz Katmanı çalışıyor...", expanded=True) as status_box:
                    status_container = st.empty()

                    def update_status(step_type: str, text: str):
                        status_container.markdown(f"**{text}**")

                    chat_history = [
                        {"role": m["role"], "content": m.get("content", "")}
                        for m in st.session_state.messages[:-1]
                    ]

                    step_result, raw_resp = st.session_state.agent.execute_react_cycle(
                        user_message=prompt,
                        chat_history=chat_history,
                        datasets=st.session_state.datasets,
                        active_key=st.session_state.active_dataset,
                        status_callback=update_status,
                    )

                    status_box.update(
                        label=f"✅ {step_result.detected_intent} ile Analiz Tamamlandı",
                        state="complete",
                        expanded=False,
                    )

                # Açıklama metni
                if step_result.explanation:
                    st.markdown(step_result.explanation)

                # Guardrail Uyarısı
                for warn in step_result.guardrail_warnings:
                    st.warning(warn)

                # Grafikler
                if step_result.has_figure:
                    for fig in step_result.figures:
                        st.plotly_chart(fig, width="stretch")

                # Teknik Detaylar, Kod ve Yönetişim
                has_code = bool(step_result.executed_code or step_result.sql_query)
                if step_result.has_applied_metrics or has_code or step_result.has_table:
                    with st.expander("🔍 Teknik Detayları ve Çalıştırılan Kodu Göster", expanded=False):
                        if step_result.has_applied_metrics:
                            st.markdown("##### 🛡️ Uygulanan Şirket İş Kuralları (Data Governance)")
                            for m in step_result.applied_metrics:
                                st.markdown(f"• **{m.canonical_name}** `v{m.version}` • *{m.owner}*")
                                st.caption(f"**Tanım:** {m.business_definition}")
                                st.code(f"Formül: {m.sql_formula}\nFiltre: {m.mandatory_filters or 'Yok'}", language="sql")
                                st.markdown("---")

                        if has_code:
                            if step_result.code_type == "python":
                                st.markdown("##### 🐍 Çalıştırılan Pandas / Python Kodu")
                                st.code(step_result.executed_code, language="python")
                            else:
                                st.markdown("##### 💾 Çalıştırılan SQL Sorgusu")
                                st.code(step_result.executed_code or step_result.sql_query, language="sql")

                        if step_result.has_table:
                            st.markdown(f"##### 📊 Sonuç Tablosu ({len(step_result.result_df):,} kayıt)")
                            st.dataframe(step_result.result_df, width="stretch")

                            csv_bytes = step_result.result_df.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                label="📥 Bu Tabloyu İndir (CSV)",
                                data=csv_bytes,
                                file_name="sql_sorgu_sonucu.csv",
                                mime="text/csv",
                                key=f"dl_active_{uuid.uuid4().hex[:6]}",
                            )

                # Hata
                if step_result.error:
                    st.error(step_result.error)

                # Asistan mesajını session'a kaydet (metrikleri dict formatında serileştirerek)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": step_result.explanation or raw_resp,
                        "sql_query": step_result.sql_query,
                        "result_df": step_result.result_df,
                        "figures": step_result.figures,
                        "plot_code": step_result.plot_code,
                        "stdout": step_result.stdout,
                        "applied_metrics": [m.to_dict() for m in step_result.applied_metrics],
                        "guardrail_warnings": step_result.guardrail_warnings,
                        "healing_notes": step_result.healing_notes,
                        "error": step_result.error,
                    }
                )

                _save_current_session()

        st.rerun()
