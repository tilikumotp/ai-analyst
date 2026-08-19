"""
Database Loader & Manager — CSV/Excel verilerini SQLite veritabanına dönüştürme ve güvenli SQL yönetimi.

Özellikler:
- CSV ve Excel DataFrame'lerini optimize edilmiş SQLite tablolarına aktarma
- Dynamic Schema Injection: PRAGMA table_info ve örnek satırlar ile semantik şema üretimi
- Defense in Depth: Read-only güvenlik denetimi (Sadece SELECT ve WITH izinli)
- Çoklu tablo desteği ve otomatik indeksleme
"""

import re
import sqlite3
import logging
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


def sanitize_table_name(name: str) -> str:
    """Dosya adını geçerli ve güvenli bir SQL tablo adına dönüştür."""
    # Uzantıyı kaldır
    clean = name.rsplit(".", 1)[0] if "." in name else name
    # Türkçe ve özel karakterleri normalize et
    clean = re.sub(r"[^a-zA-Z0-9_çÇğĞıİöÖşŞüÜ]", "_", clean)
    clean = clean.replace("ç", "c").replace("Ç", "c")\
                 .replace("ğ", "g").replace("Ğ", "g")\
                 .replace("ı", "i").replace("İ", "i")\
                 .replace("ö", "o").replace("Ö", "o")\
                 .replace("ş", "s").replace("Ş", "s")\
                 .replace("ü", "u").replace("Ü", "u")
    # Sayı ile başlıyorsa başa t_ ekle
    if clean and clean[0].isdigit():
        clean = "t_" + clean
    # Çift alt çizgileri teke indir
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean.lower() if clean else "table_data"


def sanitize_column_name(name: str) -> str:
    """Kolon adlarını temiz ve SQL uyumlu hale getir."""
    clean = str(name).strip()
    clean = re.sub(r"[^\w\s]", "_", clean)
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean if clean else "kolon"


class DatabaseManager:
    """
    SQLite Veritabanı Yöneticisi — Thread-safe in-memory/file-based SQLite veritabanı.
    """

    # Engellenen tehlikeli SQL anahtar kelimeleri (Sadece SELECT ve WITH izinlidir)
    FORBIDDEN_SQL_KEYWORDS = frozenset({
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
        "TRUNCATE", "REPLACE", "EXEC", "EXECUTE", "ATTACH", "DETACH",
        "GRANT", "REVOKE", "VACUUM", "PRAGMA", "COMMIT", "ROLLBACK"
    })

    def __init__(self, db_path: str = ":memory:"):
        """
        DatabaseManager başlatıcı.
        
        Args:
            db_path: SQLite veritabanı yolu (varsayılan: ':memory:')
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Performans için WAL modu (bellek veya dosya için hızlı okuma)
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        self._table_map: Dict[str, str] = {}  # {orijinal_dosya_adi: sql_tablo_adi}

    def load_dataframe(
        self, df: pd.DataFrame, file_name: str, table_name: Optional[str] = None
    ) -> str:
        """
        Bir DataFrame'i SQLite tablosu olarak yükle.

        Args:
            df: Yüklenecek pandas DataFrame
            file_name: Orijinal dosya adı
            table_name: Opsiyonel özel tablo adı

        Returns:
            Oluşturulan SQL tablo adı
        """
        tbl_name = table_name or sanitize_table_name(file_name)

        # Kolon isimlerini SQL dostu yap ve bir kopyasını al
        clean_df = df.copy()
        clean_df.columns = [sanitize_column_name(col) for col in clean_df.columns]

        # Tarih ve saat tiplerini string formatına çevir (SQLite uyumu)
        for col in clean_df.columns:
            if pd.api.types.is_datetime64_any_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        # SQLite'a aktar
        clean_df.to_sql(
            name=tbl_name,
            con=self.conn,
            if_exists="replace",
            index=False,
        )

        self._table_map[file_name] = tbl_name
        logger.info(f"Tablo yüklendi: '{tbl_name}' ({len(clean_df)} satır, {len(clean_df.columns)} kolon)")
        return tbl_name

    def remove_table(self, file_name_or_table: str) -> None:
        """Tabloyu veritabanından kaldır."""
        tbl_name = self._table_map.get(file_name_or_table, file_name_or_table)
        try:
            self.conn.execute(f"DROP TABLE IF EXISTS {tbl_name};")
            self._table_map = {k: v for k, v in self._table_map.items() if v != tbl_name}
        except Exception as e:
            logger.error(f"Tablo silme hatası ({tbl_name}): {e}")

    def clear_all(self) -> None:
        """Tüm tabloları sıfırla."""
        tables = self.get_table_names()
        for tbl in tables:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {tbl};")
            except Exception:
                pass
        self._table_map.clear()

    def get_table_names(self) -> List[str]:
        """Mevcut tüm SQL tablo isimlerini listele."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables

    def get_table_columns(self, table_name: str) -> List[str]:
        """Bir tablodaki gerçek kolon isimlerini listele."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"PRAGMA table_info({table_name});")
            return [row[1] for row in cursor.fetchall()]
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────
    # Dynamic Schema Injection & Data Profiling Helpers
    # ─────────────────────────────────────────────────────────
    def get_schema_context(self) -> str:
        """
        LLM için Zenginleştirilmiş Dinamik Semantik Şema (Dynamic Schema Injection) oluşturur.
        
        Her tablo için:
        - Kolon adları, SQLite tipleri
        - Kategorik kolonlar için örnek tekil değerler (['Hyundai', 'Ford', ...])
        - Sayısal kolonlar için min / max değerleri
        - Toplam satır sayısı ve ilk 3 satır örnek veri
        """
        tables = self.get_table_names()
        if not tables:
            return "Veritabanında henüz yüklü bir tablo bulunmamaktadır."

        schema_parts = []
        schema_parts.append(f"Veritabanında toplam {len(tables)} tablo bulunmaktadır:\n")

        for tbl in tables:
            cursor = self.conn.cursor()

            # 1) Toplam satır sayısı
            cursor.execute(f"SELECT COUNT(*) FROM {tbl};")
            total_rows = cursor.fetchone()[0]

            # 2) Kolon bilgileri
            cursor.execute(f"PRAGMA table_info({tbl});")
            columns_info = cursor.fetchall()
            # columns_info: [(cid, name, type, notnull, dflt_value, pk), ...]

            col_descriptions = []
            for col in columns_info:
                col_name = col[1]
                col_type = col[2] or "TEXT"
                
                # Zenginleştirilmiş kolon detayı: distinct değerler veya min/max
                detail_str = ""
                try:
                    # Metin / kategorik ise en sık geçen birkaç tekil değeri göster
                    if "INT" in col_type.upper() or "REAL" in col_type.upper() or "FLOAT" in col_type.upper() or "NUM" in col_type.upper():
                        cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}), ROUND(AVG({col_name}), 2) FROM {tbl} WHERE {col_name} IS NOT NULL;")
                        min_v, max_v, avg_v = cursor.fetchone()
                        if min_v is not None:
                            detail_str = f" [Min: {min_v}, Max: {max_v}, Ort: {avg_v}]"
                    else:
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {tbl} WHERE {col_name} IS NOT NULL LIMIT 6;")
                        distinct_samples = [str(r[0]) for r in cursor.fetchall() if str(r[0]).strip()]
                        if distinct_samples:
                            sample_preview = ", ".join(f"'{s}'" for s in distinct_samples[:5])
                            detail_str = f" [Örnek Değerler: {sample_preview}]"
                except Exception:
                    pass

                col_descriptions.append(f"    - `{col_name}` ({col_type}){detail_str}")

            # 3) Örnek ilk 3 satır
            cursor.execute(f"SELECT * FROM {tbl} LIMIT 3;")
            sample_rows = cursor.fetchall()
            col_names = [col[1] for col in columns_info]

            sample_df = pd.DataFrame(sample_rows, columns=col_names)
            sample_str = sample_df.to_string(index=False)

            table_block = f"""━━━ 📊 Tablo: `{tbl}` ({total_rows:,} satır) ━━━
Kolonlar ve Değer Aralıkları:
{chr(10).join(col_descriptions)}

Örnek Veriler (İlk 3 Satır):
{sample_str}
"""
            schema_parts.append(table_block)

        return "\n\n".join(schema_parts)

    def get_column_profile(
        self, table_name: str, column_name: str, top_n: int = 15
    ) -> pd.DataFrame:
        """
        Belirli bir kolonun frekans dağılımını (value counts) ve yüzdelerini deterministik hesaplar.
        """
        clean_tbl = sanitize_table_name(table_name)
        clean_col = sanitize_column_name(column_name)
        
        query = f"""
        SELECT 
            {clean_col} AS Deger,
            COUNT(*) AS Adet,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {clean_tbl}), 2) AS Yuzde
        FROM {clean_tbl}
        GROUP BY {clean_col}
        ORDER BY Adet DESC
        LIMIT {int(top_n)};
        """
        return self.execute_query(query)

    def read_sample_rows(
        self, table_name: str, limit: int = 10, offset: int = 0
    ) -> pd.DataFrame:
        """Tablodan belirli sayıda satır oku."""
        clean_tbl = sanitize_table_name(table_name)
        query = f"SELECT * FROM {clean_tbl} LIMIT {int(limit)} OFFSET {int(offset)};"
        return self.execute_query(query)

    # ─────────────────────────────────────────────────────────
    # Güvenli SQL Çalıştırma
    # ─────────────────────────────────────────────────────────
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        SQL sorgusunu salt-okunur güvenlik denetiminden geçirip çalıştırır.

        Args:
            query: Çalıştırılacak SQL sorgusu

        Returns:
            pd.DataFrame olarak sorgu sonucu

        Raises:
            PermissionError: Eğer sorgu SELECT/WITH dışında bir işlem yapmaya çalışırsa
            sqlite3.Error: SQL sözdizimi veya çalışma hatası durumunda
        """
        clean_query = query.strip()

        # Yorumları ve baştaki boşlukları temizle
        clean_query_no_comments = re.sub(r"--.*$", "", clean_query, flags=re.MULTILINE)
        clean_query_no_comments = re.sub(r"/\*.*?\*/", "", clean_query_no_comments, flags=re.DOTALL).strip()

        # 1) Sadece SELECT veya WITH ile başlamalı
        first_token = clean_query_no_comments.split()[0].upper() if clean_query_no_comments else ""
        if first_token not in ("SELECT", "WITH", "EXPLAIN"):
            raise PermissionError(
                f"Güvenlik İhlali: Sadece okuma amaçlı (SELECT / WITH) sorgulara izin verilir. '{first_token}' engellendi."
            )

        # 2) Engellenen anahtar kelimeleri kelime sınırlarıyla kontrol et
        upper_query = clean_query_no_comments.upper()
        for forbidden in self.FORBIDDEN_SQL_KEYWORDS:
            # Kelime sınırı ile ara (örn: DROP TABLE engellenir ama 'dropdown' kolonu etkilenmez)
            if re.search(rf"\b{forbidden}\b", upper_query):
                raise PermissionError(
                    f"Güvenlik İhlali: SQL sorgusu yasaklı '{forbidden}' anahtar kelimesini içeriyor."
                )

        # 3) Sorguyu çalıştır ve DataFrame olarak döndür
        try:
            result_df = pd.read_sql_query(clean_query, self.conn)
            return result_df
        except sqlite3.Error as e:
            logger.warning(f"SQL Çalıştırma Hatası: {e}\nSorgu: {clean_query}")
            raise e
        except Exception as e:
            logger.error(f"Beklenmeyen SQL Hatası: {e}")
            raise e
