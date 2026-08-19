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
import uuid
from datetime import datetime
from typing import Optional


SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions")


class ChatHistory:
    """JSON tabanlı sohbet geçmişi yöneticisi."""

    def __init__(self):
        os.makedirs(SESSIONS_DIR, exist_ok=True)

    def save_session(
        self,
        session_id: str,
        messages: list[dict],
        file_names: Optional[list[str]] = None,
        title: Optional[str] = None,
    ) -> str:
        """
        Sohbet oturumunu kaydet veya güncelle.

        Args:
            session_id: Oturum kimliği
            messages: Mesaj listesi [{"role": ..., "content": ...}, ...]
            file_names: Yüklü dosya adları listesi
            title: Oturum başlığı (None ise ilk kullanıcı mesajından oluşturulur)

        Returns:
            Oturum ID'si
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

        # Mesajları kaydet (sadece role ve content — figürler saklanamaz)
        clean_messages = []
        for msg in messages:
            clean_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            })

        session["messages"] = clean_messages
        session["updated_at"] = datetime.now().isoformat()
        session["file_names"] = file_names or []
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

        Returns:
            [{"id": ..., "title": ..., "updated_at": ..., "message_count": ..., "file_names": ...}, ...]
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
                    # Geriye uyumluluk
                    "file_name": file_names[0] if file_names else None,
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # En yeni önce
        sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Bir oturumu sil."""
        filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    @staticmethod
    def generate_id() -> str:
        """Yeni oturum ID'si oluştur."""
        return uuid.uuid4().hex[:12]
