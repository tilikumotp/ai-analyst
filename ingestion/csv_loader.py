"""
Data Loader — Akıllı encoding ve separator algılama ile CSV/Excel yükleme.

Desteklenen formatlar:
- CSV (.csv) — Otomatik encoding (UTF-8, UTF-8-BOM, CP1254, ISO-8859-9, Latin-1, CP1252) ve separator algılama
- Excel (.xlsx, .xls) — Birden fazla sheet desteği

Türkçe dosyalar için yaygın encoding'leri (cp1254, iso-8859-9) otomatik algılar.
Kolon tiplerini (tarih, sayı, kategori) veri kaybı veya sayı bozulması olmadan akıllıca dönüştürür.
"""

import csv
import io
import re
from typing import Optional, Tuple
import chardet
import pandas as pd


class CSVLoader:
    """CSV ve Excel dosyalarını akıllıca yükler — encoding, separator ve tip algılama."""

    ENCODINGS = ["utf-8-sig", "utf-8", "cp1254", "iso-8859-9", "latin-1", "cp1252"]
    SEPARATORS = [",", ";", "\t", "|"]

    @classmethod
    def load(
        cls, uploaded_file, sheet_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Streamlit UploadedFile nesnesinden DataFrame yükle.

        Args:
            uploaded_file: Streamlit'in st.file_uploader'dan döndürdüğü dosya nesnesi
            sheet_name: Excel dosyaları için sheet adı (None ise ilk sheet)

        Returns:
            (DataFrame, metadata_dict) tuple'ı
        """
        file_name = uploaded_file.name.lower()

        if file_name.endswith((".xlsx", ".xls")):
            return cls._load_excel(uploaded_file, sheet_name)
        else:
            return cls._load_csv(uploaded_file)

    @classmethod
    def get_sheet_names(cls, uploaded_file) -> list[str]:
        """Excel dosyasındaki sheet isimlerini döndür."""
        try:
            xls = pd.ExcelFile(uploaded_file)
            return xls.sheet_names
        except Exception:
            return []

    # ──────────────────────────────────────────────────
    # CSV Yükleme
    # ──────────────────────────────────────────────────

    @classmethod
    def _load_csv(cls, uploaded_file) -> Tuple[pd.DataFrame, dict]:
        """CSV dosyası yükle."""
        raw_bytes = uploaded_file.getvalue()

        # Encoding algıla
        encoding = cls._detect_encoding(raw_bytes)

        # Text'e çevir
        text = raw_bytes.decode(encoding, errors="replace")

        # Separator algıla
        separator = cls._detect_separator(text)

        # DataFrame oluştur - önce c motoru sonra python motoru dene
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=separator,
                on_bad_lines="skip",
                engine="c",
            )
        except Exception:
            df = pd.read_csv(
                io.StringIO(text),
                sep=separator,
                on_bad_lines="skip",
                engine="python",
            )

        # Boş kolon isimlerini düzelt ve temizle
        clean_cols = []
        for i, col in enumerate(df.columns):
            c_str = str(col).strip()
            if not c_str or c_str.startswith("Unnamed"):
                clean_cols.append(f"Kolon_{i+1}")
            else:
                clean_cols.append(c_str)
        df.columns = clean_cols

        # Otomatik tip dönüşümü
        df = cls._auto_convert_types(df)

        # Metadata
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        metadata = {
            "dosya_adi": uploaded_file.name,
            "dosya_tipi": "CSV",
            "encoding": encoding,
            "separator": repr(separator),
            "satir_sayisi": len(df),
            "kolon_sayisi": len(df.columns),
            "bellek_kullanimi": f"{memory_mb:.2f} MB",
        }

        return df, metadata

    # ──────────────────────────────────────────────────
    # Excel Yükleme
    # ──────────────────────────────────────────────────

    @classmethod
    def _load_excel(
        cls, uploaded_file, sheet_name: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """Excel dosyası yükle (.xlsx, .xls)."""
        try:
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names

            # Sheet seçimi
            if sheet_name and sheet_name in sheet_names:
                selected_sheet = sheet_name
            else:
                selected_sheet = sheet_names[0]  # İlk sheet

            df = pd.read_excel(xls, sheet_name=selected_sheet)

        except Exception as e:
            raise ValueError(f"Excel dosyası okunamadı: {e}")

        # Boş kolon isimlerini düzelt
        clean_cols = []
        for i, col in enumerate(df.columns):
            c_str = str(col).strip()
            if not c_str or c_str.startswith("Unnamed"):
                clean_cols.append(f"Kolon_{i+1}")
            else:
                clean_cols.append(c_str)
        df.columns = clean_cols

        # Otomatik tip dönüşümü
        df = cls._auto_convert_types(df)

        # Metadata
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        metadata = {
            "dosya_adi": uploaded_file.name,
            "dosya_tipi": "Excel",
            "encoding": "N/A",
            "separator": "N/A",
            "sheet": selected_sheet,
            "toplam_sheet": len(sheet_names),
            "sheet_listesi": sheet_names,
            "satir_sayisi": len(df),
            "kolon_sayisi": len(df.columns),
            "bellek_kullanimi": f"{memory_mb:.2f} MB",
        }

        return df, metadata

    # ──────────────────────────────────────────────────
    # Yardımcı Metodlar
    # ──────────────────────────────────────────────────

    @classmethod
    def _detect_encoding(cls, raw_bytes: bytes) -> str:
        """Dosyanın encoding'ini otomatik ve güvenli algıla (Türkçe karakter koruması)."""
        sample = raw_bytes[:65536]

        # 1) UTF-8 / UTF-8-SIG strict kontrolü
        try:
            sample.decode("utf-8")
            return "utf-8"
        except UnicodeDecodeError:
            pass

        # 2) chardet algılama
        result = chardet.detect(sample)
        if result.get("confidence", 0) > 0.75 and result.get("encoding"):
            detected = result["encoding"].lower()
            encoding_map = {
                "ascii": "utf-8",
                "windows-1254": "cp1254",
                "iso-8859-9": "cp1254",
                "windows-1252": "cp1252",
            }
            cand = encoding_map.get(detected, detected)
            try:
                sample.decode(cand)
                return cand
            except Exception:
                pass

        # 3) Sırayla Türkçe ve Batı Avrupa encoding'lerini dene
        for enc in ["cp1254", "iso-8859-9", "latin-1", "cp1252", "utf-8-sig"]:
            try:
                sample.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue

        return "utf-8"

    @classmethod
    def _detect_separator(cls, text: str) -> str:
        """CSV separator'ını otomatik algıla."""
        # İlk 10 dolu satırı al
        lines = [line for line in text.splitlines() if line.strip()][:10]
        if not lines:
            return ","

        sample_text = "\n".join(lines)

        # 1) csv.Sniffer ile dene
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters=",;\t|")
            if dialect and dialect.delimiter in cls.SEPARATORS:
                return dialect.delimiter
        except Exception:
            pass

        # 2) Fallback frekans ve satır tutarlılığı analizi
        header = lines[0]
        scores = {}
        for sep in cls.SEPARATORS:
            header_count = header.count(sep)
            if header_count > 0:
                # Satırlar arası tutarlılık
                consistent = all(line.count(sep) == header_count for line in lines[1:5])
                scores[sep] = header_count * (2.0 if consistent else 1.0)

        if scores:
            return max(scores, key=scores.get)

        return ","

    @classmethod
    def _auto_convert_types(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kolon tiplerini veri kaybı veya sayı bozulması olmadan dönüştür.
        - Sayısal kolonlar (örn: "4500.00", "15", "-3.5"): float/int
        - Türkçe formatlı sayılar ("1.234,56"): kontrollü float
        - Tarih formatları: datetime
        - Model veya alfanümerik metinler ("1.4 TSI", "320d"): METİN olarak korunur.
        """
        for col in df.columns:
            if df[col].dtype != "object":
                continue

            non_null = df[col].dropna().astype(str).str.strip()
            if len(non_null) == 0:
                continue

            # 1) Standart Sayısal Dönüşüm (Örn: "4500.00", "1250", "-3.14")
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().sum() >= len(non_null) * 0.9:
                    df[col] = numeric
                    continue
            except Exception:
                pass

            # 2) Türkçe / Avrupa Formatlı Sayılar (Örn: "1.234,56" veya "1234,56")
            try:
                turkish_num_pattern = r"^\s*-?\d{1,3}(?:\.\d{3})*,\d+\s*$"
                is_turkish_numeric = non_null.str.match(turkish_num_pattern).mean() >= 0.8
                if is_turkish_numeric:
                    cleaned = (
                        df[col]
                        .astype(str)
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False)
                    )
                    numeric = pd.to_numeric(cleaned, errors="coerce")
                    if numeric.notna().sum() >= len(non_null) * 0.8:
                        df[col] = numeric
                        continue
            except Exception:
                pass

            # 3) Tarih Dönüşümü
            try:
                date_like_sample = non_null.head(20)
                has_date_delimiters = date_like_sample.str.contains(r"[\-/\.]\d{2,4}", regex=True).mean() > 0.8
                if has_date_delimiters:
                    converted = pd.to_datetime(df[col], format="mixed", errors="coerce")
                    if converted.notna().sum() >= len(non_null) * 0.8:
                        df[col] = converted
                        continue
            except Exception:
                pass

            # 4) Kategorik Dönüşüm
            unique_count = df[col].nunique()
            if unique_count <= 30 and (unique_count / max(len(df), 1)) < 0.05:
                df[col] = df[col].astype("category")

        return df
