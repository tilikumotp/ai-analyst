"""
Data Loader — Akıllı encoding ve separator algılama ile CSV/Excel yükleme.

Desteklenen formatlar:
- CSV (.csv) — Otomatik encoding (UTF-8, cp1254, Latin-1, ...) ve separator algılama
- Excel (.xlsx, .xls) — Birden fazla sheet desteği

Türkçe dosyalar için yaygın encoding'leri (cp1254, iso-8859-9) otomatik algılar.
Kolon tiplerini (tarih, sayı, kategori) otomatik dönüştürür.
"""

import io
import pandas as pd
import chardet
from typing import Tuple, Optional


class CSVLoader:
    """CSV ve Excel dosyalarını akıllıca yükler — encoding, separator ve tip algılama."""

    # Türkçe dosyalar için yaygın encoding'ler
    ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1254", "iso-8859-9", "cp1252"]
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

        # DataFrame oluştur
        df = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            on_bad_lines="skip",
            engine="python",
        )

        # Boş kolon isimlerini düzelt
        df.columns = [
            f"Kolon_{i}" if str(col).strip() == "" else str(col).strip()
            for i, col in enumerate(df.columns)
        ]

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
        df.columns = [
            f"Kolon_{i}" if str(col).strip() == "" or str(col).startswith("Unnamed")
            else str(col).strip()
            for i, col in enumerate(df.columns)
        ]

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
        """Dosyanın encoding'ini otomatik algıla."""
        # chardet ile otomatik algılama (ilk 10KB yeterli)
        result = chardet.detect(raw_bytes[:10000])

        if result["confidence"] > 0.7 and result["encoding"]:
            detected = result["encoding"].lower()

            # Yaygın mapping düzeltmeleri
            encoding_map = {
                "ascii": "utf-8",
                "windows-1254": "cp1254",
                "iso-8859-9": "cp1254",
                "windows-1252": "cp1252",
            }
            return encoding_map.get(detected, detected)

        # Düşük güven → sırayla dene
        for enc in cls.ENCODINGS:
            try:
                raw_bytes.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue

        return "utf-8"  # Son çare

    @classmethod
    def _detect_separator(cls, text: str) -> str:
        """CSV separator'ını otomatik algıla."""
        first_lines = text.split("\n")[:10]
        header = first_lines[0] if first_lines else ""

        scores = {}
        for sep in cls.SEPARATORS:
            count = header.count(sep)
            if count > 0:
                # Tutarlılık: ilk satırlarda aynı sayıda mı?
                consistent = all(
                    line.count(sep) == count
                    for line in first_lines[1:5]
                    if line.strip()
                )
                scores[sep] = count if consistent else count * 0.5

        if scores:
            return max(scores, key=scores.get)
        return ","  # Varsayılan

    @classmethod
    def _auto_convert_types(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Kolon tiplerini akıllıca dönüştür (tarih, sayı, kategori)."""
        for col in df.columns:
            if df[col].dtype != "object":
                continue

            # 1) Tarih dönüşümü
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True)
                # En az %50'si valid tarih mi?
                if converted.notna().sum() > len(df) * 0.5:
                    df[col] = converted
                    continue
            except (ValueError, TypeError):
                pass

            # 2) Sayısal dönüşüm (Türkçe virgüllü ondalık: "1.234,56" → 1234.56)
            try:
                if df[col].str.contains(r"^\s*-?\d", na=False).mean() > 0.5:
                    cleaned = (
                        df[col]
                        .str.replace(".", "", regex=False)
                        .str.replace(",", ".", regex=False)
                    )
                    numeric = pd.to_numeric(cleaned, errors="coerce")
                    # En az %80'i valid sayı mı?
                    if numeric.notna().sum() > len(df) * 0.8:
                        df[col] = numeric
                        continue
            except (ValueError, TypeError, AttributeError):
                pass

            # 3) Kategorik dönüşüm (az benzersiz değer varsa)
            unique_ratio = df[col].nunique() / max(len(df), 1)
            if unique_ratio < 0.05 and df[col].nunique() < 50:
                df[col] = df[col].astype("category")

        return df
