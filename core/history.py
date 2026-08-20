"""
Chat History — JSON tabanlı sohbet geçmişi yönetimi.

Her sohbet oturumu ayrı bir JSON dosyası olarak saklanır:
data/sessions/{session_id}.json

Yapı:
{
    "id": "abc123",
    "title": "Satış verisi analizi",
    "created_at": "2024-01-15T10:30:00",
    "updated_at": "2024-01-15T11:00:00",
    "file_names": ["satis_verisi.csv", "musteri.xlsx"],
    "messages": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
    ]
}

Not: Plotly figürleri ve DataFrame'ler JSON'a kaydedilemez.
Sadece metin içerikleri saklanır.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd


SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions")


class ChatHistory:
    """JSON ve CSV tabanlı kalıcı sohbet ve veri seti geçmişi yöneticisi."""

    def __init__(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        os.makedirs(os.path.join(SESSIONS_DIR, "datasets"), exist_ok=True)

    def save_session(
        self,
        session_id: str,
        messages: list[dict],
        file_names: Optional[list[str]] = None,
        title: Optional[str] = None,
        active_dataset: Optional[str] = None,
    ) -> str:
        """
        Sohbet oturumunu kaydet veya güncelle.
        """
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")

        # Mevcut oturum varsa yükle
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                session = json.load(f)
        else:
            session = {
                "id": session_id,
                "created_at": datetime.now().isoformat(),
            }

        # Başlık belirleme
        if title:
            session["title"] = title
        elif "title" not in session or not session.get("title"):
            # İlk kullanıcı mesajından başlık oluştur
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                first_msg = user_msgs[0]["content"]
                session["title"] = first_msg[:60] + ("..." if len(first_msg) > 60 else "")
            else:
                session["title"] = "Yeni Sohbet"

        # Mesajları kaydet
        clean_messages = []
        for msg in messages:
            clean_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "sql_query": msg.get("sql_query"),
                "executed_code": msg.get("executed_code"),
                "code_type": msg.get("code_type", "python"),
                "applied_metrics": msg.get("applied_metrics", []),
                "guardrail_warnings": msg.get("guardrail_warnings", []),
                "critic_notes": msg.get("critic_notes", []),
                "lineage_mermaid": msg.get("lineage_mermaid"),
                "is_cached": msg.get("is_cached", False),
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            })

        session["messages"] = clean_messages
        session["updated_at"] = datetime.now().isoformat()
        session["file_names"] = file_names or []
        session["active_dataset"] = active_dataset
        session["message_count"] = len(clean_messages)

        # Geriye uyumluluk: eski file_name alanını da güncelle
        if file_names:
            session["file_name"] = file_names[0]
        else:
            session["file_name"] = None

        # Dosyaya yaz
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

        return session_id

    def save_datasets(self, session_id: str, datasets: Dict[str, Any]) -> None:
        """Oturuma ait veri setlerini (DataFrame ve Metadata) diske kalıcı olarak kaydet."""
        if not datasets:
            return

        session_data_dir = os.path.join(SESSIONS_DIR, "datasets", session_id)
        os.makedirs(session_data_dir, exist_ok=True)

        for key, entry in datasets.items():
            if not isinstance(entry, dict) or "df" not in entry:
                continue

            df: pd.DataFrame = entry["df"]
            metadata: dict = entry.get("metadata", {})
            sql_table: str = entry.get("sql_table", "")

            # Güvenli dosya adı
            safe_key = "".join(c if c.isalnum() or c in (".", "_", "-") else "_" for c in key)
            csv_path = os.path.join(session_data_dir, f"{safe_key}.csv")
            meta_path = os.path.join(session_data_dir, f"{safe_key}.meta.json")

            # DataFrame'i CSV olarak sakla
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")

            # Metadata'yı JSON olarak sakla
            meta_payload = {
                "key": key,
                "metadata": metadata,
                "sql_table": sql_table,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, ensure_ascii=False, indent=2)

    def load_datasets(self, session_id: str) -> Dict[str, Any]:
        """Oturuma ait kayıtlı veri setlerini diskten yükle."""
        datasets = {}
        session_data_dir = os.path.join(SESSIONS_DIR, "datasets", session_id)
        if not os.path.exists(session_data_dir):
            return datasets

        for fname in os.listdir(session_data_dir):
            if fname.endswith(".meta.json"):
                meta_path = os.path.join(session_data_dir, fname)
                base_name = fname[:-10]  # .meta.json kısmını kaldır
                csv_path = os.path.join(session_data_dir, f"{base_name}.csv")

                if os.path.exists(csv_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta_payload = json.load(f)

                        df = pd.read_csv(csv_path, encoding="utf-8-sig")
                        key = meta_payload.get("key", base_name)
                        datasets[key] = {
                            "df": df,
                            "metadata": meta_payload.get("metadata", {}),
                            "sql_table": meta_payload.get("sql_table", ""),
                        }
                    except Exception:
                        continue

        return datasets

    def load_session(self, session_id: str) -> Optional[dict]:
        """Bir oturumu yükle."""
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if not os.path.exists(filepath):
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Geriye uyumluluk: eski file_name → file_names
        if "file_names" not in data:
            old_name = data.get("file_name")
            data["file_names"] = [old_name] if old_name else []

        return data

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """
        Tüm oturumları listele (en yeni önce).
        """
        sessions = []

        if not os.path.exists(SESSIONS_DIR):
            return sessions

        for filename in os.listdir(SESSIONS_DIR):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(SESSIONS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Geriye uyumluluk: eski file_name → file_names
                file_names = data.get("file_names")
                if file_names is None:
                    old_name = data.get("file_name")
                    file_names = [old_name] if old_name else []

                sessions.append({
                    "id": data.get("id", filename.replace(".json", "")),
                    "title": data.get("title", "İsimsiz Sohbet"),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": data.get("message_count", 0),
                    "file_names": file_names,
                    "active_dataset": data.get("active_dataset"),
                    # Geriye uyumluluk
                    "file_name": file_names[0] if file_names else None,
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # En yeni önce
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Bir oturumu ve diske kaydedilmiş veri setlerini sil."""
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

        session_data_dir = os.path.join(SESSIONS_DIR, "datasets", session_id)
        if os.path.exists(session_data_dir):
            shutil.rmtree(session_data_dir, ignore_errors=True)

        return True

    @staticmethod
    def generate_id() -> str:
        """Yeni oturum ID'si oluştur."""
        return uuid.uuid4().hex[:12]
