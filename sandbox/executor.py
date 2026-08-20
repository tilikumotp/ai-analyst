"""
Safe Executor — LLM'in ürettiği Python kodunu güvenli ortamda çalıştırır.

Güvenlik önlemleri:
- Tehlikeli modüller engellenir (os, sys, subprocess, ...)
- __builtins__ kısıtlanır (sadece güvenli built-in'ler)
- Import'lar whitelist ile kontrol edilir (fromlist de kontrol edilir)
- AST tabanlı yapısal tarama (eval, exec, open çağrıları)
- Zaman aşımı (timeout) uygulanır
- stdout yakalanır
- Plotly figure ve DataFrame sonuçları otomatik algılanır
- Dosya yazma işlemleri engellenir (plotly.io bloklanır)

Çoklu veri seti desteği:
- df → aktif veri seti
- datasets["key"] → isimle erişim
- df_<isim> → kısa yol değişkenleri

NOT: Thread-based timeout kullanılır. Python'da thread'ler zorla durdurulamaz;
timeout aşılırsa sonuç alınamaz ve TimeoutError fırlatılır. Thread arka planda
çalışmaya devam edebilir (best-effort timeout).
"""

import ast
import io
import re
import sys
import threading
import traceback
import builtins as _builtins
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subplots

# Matplotlib & Seaborn desteği (Headless backend ile güvenli çalıştırma)
try:
    import matplotlib
    matplotlib.use("Agg")  # GUI thread çökmesini önlemek için headless backend
    import matplotlib.pyplot as plt
    import matplotlib.figure
except ImportError:
    matplotlib = None
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None


# ─────────────────────────────────────────────────────────
# Güvenli Import Mekanizması
# ─────────────────────────────────────────────────────────

# Import'a izin verilen temel modüller (ve alt modülleri)
ALLOWED_BASE_MODULES = frozenset({
    # Veri bilimi & görselleştirme
    "pandas", "numpy", "plotly", "matplotlib", "seaborn", "scipy", "sklearn",
    # Matematik & istatistik
    "math", "statistics", "decimal", "fractions",
    # Koleksiyonlar & araçlar
    "collections", "itertools", "functools", "operator",
    # Metin & tarih
    "string", "textwrap", "datetime", "re",
    # Veri formatları
    "json", "csv",
    # Yardımcı
    "copy", "typing", "dataclasses", "enum",
})

# Güvensiz alt modüller / import yolları (explicit engelleme)
BLOCKED_IMPORT_PATHS = frozenset({
    "plotly.io",
    "numpy.core.os",
    "matplotlib.pyplot.show",  # blocking GUI show engeli
})

_original_import = _builtins.__import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    Sadece whitelist'teki modüllerin import edilmesine izin verir.

    fromlist içindeki öğeler de kontrol edilir (örn. from numpy import os engellenir).
    """
    # Göreli import'ları engelle (level > 0)
    if level > 0:
        raise ImportError(
            f"Göreli import'lar (level={level}) güvenlik nedeniyle engellenmiştir."
        )

    base_module = name.split(".")[0]

    # Temel modül kontrolü
    if base_module not in ALLOWED_BASE_MODULES:
        raise ImportError(
            f"'{name}' modülü güvenlik nedeniyle engellenmiştir. "
            f"Kullanabileceğin modüller: {', '.join(sorted(ALLOWED_BASE_MODULES))}"
        )

    # Bloklanmış alt yollar (örn. plotly.io → dosya yazma)
    full_path = name
    if fromlist:
        for sub in fromlist:
            if sub in ("os", "sys", "subprocess", "shutil", "socket", "pathlib"):
                raise ImportError(
                    f"'{name}.{sub}' güvenlik nedeniyle engellenmiştir."
                )

    if full_path in BLOCKED_IMPORT_PATHS:
        raise ImportError(
            f"'{full_path}' modülü güvenlik nedeniyle (dosya I/O riski) engellenmiştir."
        )

    return _original_import(name, globals, locals, fromlist, level)


def _sanitize_key(name: str) -> str:
    """Dosya adını güvenli Python değişken adına dönüştür."""
    key = name.rsplit(".", 1)[0] if "." in name else name
    key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
    if key and key[0].isdigit():
        key = "_" + key
    key = re.sub(r'_+', '_', key).strip('_')
    return key.lower()


@dataclass
class ExecutionResult:
    """Kod çalıştırma sonucu."""

    stdout: str = ""
    figures: list = field(default_factory=list)
    result_df: Optional[pd.DataFrame] = None
    modified_df: Optional[pd.DataFrame] = None
    # Çoklu dataset değişiklikleri: {"key": modified_df, ...}
    modified_datasets: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Kod başarıyla çalıştı mı? (agent.py ile uyumlu)"""
        return self.error is None

    @property
    def has_figure(self) -> bool:
        return len(self.figures) > 0

    @property
    def has_result_df(self) -> bool:
        return self.result_df is not None and not self.result_df.empty

    @property
    def has_error(self) -> bool:
        return self.error is not None


class SecurityError(Exception):
    """Güvenlik taramasından geçilemedi."""
    pass


class SafeExecutor:
    """LLM tarafından üretilen Python kodunu güvenli bir ortamda çalıştırır."""

    # İzin verilen built-in fonksiyonlar
    SAFE_BUILTINS = {
        # ── Import mekanizması ──
        "__import__": _safe_import,
        "__build_class__": _builtins.__build_class__,
        # ── Yazdırma ve dönüştürme ──
        "print": print,
        "repr": repr,
        "format": format,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "complex": complex,
        "bytes": bytes,
        "bytearray": bytearray,
        # ── Koleksiyonlar ──
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        # ── İteratorler ──
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "sorted": sorted,
        "reversed": reversed,
        "iter": iter,
        "next": next,
        # ── Matematik ──
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "pow": pow,
        "divmod": divmod,
        # ── Kontrol ──
        "len": len,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "type": type,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "delattr": delattr,
        "callable": callable,
        "any": any,
        "all": all,
        "hash": hash,
        "id": id,
        "dir": dir,
        "vars": vars,
        # ── Encoding ──
        "chr": chr,
        "ord": ord,
        "hex": hex,
        "oct": oct,
        "bin": bin,
        # ── Sınıf araçları ──
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "property": property,
        "super": super,
        "object": object,
        "slice": slice,
        # ── Sabitler ──
        "True": True,
        "False": False,
        "None": None,
        # ── Yaygın istisnalar ──
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
        "ZeroDivisionError": ZeroDivisionError,
        "StopIteration": StopIteration,
        "Exception": Exception,
        "RuntimeError": RuntimeError,
        "ImportError": ImportError,
        "NameError": NameError,
        "NotImplementedError": NotImplementedError,
    }

    # Maksimum desteklenen veri seti sayısı
    MAX_DATASETS = 5

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Kamu API
    # ------------------------------------------------------------------
    def execute_plot(
        self,
        code: str,
        result_df: Optional[pd.DataFrame] = None,
    ) -> ExecutionResult:
        """
        Görselleştirme kodunu doğrudan SQL sonucu (result_df) üzerinde çalıştır.

        Args:
            code: Plotly Python kodu
            result_df: SQL sorgusundan dönen DataFrame

        Returns:
            ExecutionResult — figures, stdout, error
        """
        result = ExecutionResult()
        clean_df = result_df.copy() if result_df is not None else pd.DataFrame()

        # ── 1. Güvenlik Kontrolü ──
        try:
            self._check_security(code)
        except SecurityError as exc:
            result.error = f"🛡️ Güvenlik: {exc}"
            return result

        # ── 2. Kod Ön İşleme ──
        code = self._preprocess_code(code)

        # ── 3. stdout Yakalama ──
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        # ── 4. Ortam Kurulumu ──
        exec_globals = {
            "__builtins__": self.SAFE_BUILTINS,
            "result_df": clean_df,
            "df": clean_df,  # Geriye uyumluluk için
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "make_subplots": subplots.make_subplots,
            "fig": None,
        }

        try:
            self._exec_with_timeout(code, exec_globals)
            result.stdout = buffer.getvalue()

            # Figürleri topla
            for var_name, var_value in exec_globals.items():
                if var_name.startswith("_"):
                    continue
                if var_value is not None and isinstance(var_value, go.Figure):
                    result.figures.append(var_value)

        except TimeoutError:
            result.error = f"⏱️ Zaman Aşımı: Görselleştirme {self.timeout}s içinde tamamlanamadı."
        except Exception as e:
            result.error = self._format_error(e)
        finally:
            sys.stdout = old_stdout

        return result

    def execute(
        self,
        code: str,
        datasets: dict,
        active_key: str,
    ) -> ExecutionResult:
        """
        Kodu güvenli ortamda çalıştır ve sonuçları döndür.

        Args:
            code: Çalıştırılacak Python kodu
            datasets: {"key": {"df": DataFrame, "metadata": dict}, ...}
            active_key: Aktif veri setinin anahtarı

        Returns:
            ExecutionResult — stdout, figure, result_df, modified_df, modified_datasets, error
        """
        result = ExecutionResult()

        # ── 1. Güvenlik Kontrolü ──
        try:
            self._check_security(code)
        except SecurityError as exc:
            result.error = f"🛡️ Güvenlik: {exc}"
            return result

        # ── 2. Kod Ön İşleme ──
        code = self._preprocess_code(code)

        # ── 3. stdout Yakalama ──
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        # ── 4. Çalıştırma Ortamı ──
        # Tek bir globals dict kullan (exec locals sorununu önler)
        exec_globals = self._build_exec_globals(code, datasets, active_key)

        try:
            # ── 5. Timeout ile Çalıştır (process-based) ──
            self._exec_with_timeout(code, exec_globals)

            # ── 6. Sonuçları Topla ──
            result.stdout = buffer.getvalue()
            result = self._collect_results(result, exec_globals, datasets)

        except TimeoutError:
            result.error = (
                f"⏱️ Zaman Aşımı: Kod {self.timeout} saniye içinde tamamlanamadı.\n"
                "İpucu: Daha az veriyle çalışın veya işlemi parçalara bölün."
            )
        except Exception as e:
            result.error = self._format_error(e)
        finally:
            sys.stdout = old_stdout

        return result

    # ------------------------------------------------------------------
    # Çalıştırma ortamı kurulumu
    # ------------------------------------------------------------------
    def _build_exec_globals(self, code: str, datasets: dict, active_key: str = "") -> dict:
        """
        exec() için globals dict oluştur.

        NOT: exec() locals_dict ile güvenilir değil — tüm değişkenler
        globals'e yazılır. Böylece figür ve result_df toplama kesin çalışır.
        """
        # Tüm datasetlerin kopyaları
        datasets_copy = {}
        for key, entry in datasets.items():
            datasets_copy[key] = entry["df"].copy()

        # df → aktif dataset (veya ilk yüklenen dataset kısayolu)
        if active_key and active_key in datasets_copy:
            df_copy = datasets_copy[active_key]
        else:
            first_key = next(iter(datasets_copy)) if datasets_copy else None
            df_copy = datasets_copy[first_key] if first_key else pd.DataFrame()

        exec_globals = {
            "__builtins__": self.SAFE_BUILTINS,
            # İlk DataFrame (kısayol)
            "df": df_copy,
            # Tüm datasetler (dict erişimi)
            "datasets": datasets_copy,
            # Kütüphaneler (import gerekmeden erişilebilir)
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "make_subplots": subplots.make_subplots,
            "plt": plt,
            "sns": sns,
            "matplotlib": matplotlib,
            # Sonuç değişkenleri
            "fig": None,
            "result_df": None,
        }

        # df_<isim> kısa yol değişkenleri ekle
        for key, entry in datasets.items():
            meta = entry.get("metadata", {})
            file_name = meta.get("dosya_adi", key)
            var_name = f"df_{_sanitize_key(file_name)}"
            exec_globals[var_name] = datasets_copy[key]

        return exec_globals

    # ------------------------------------------------------------------
    # Sonuç toplama
    # ------------------------------------------------------------------
    def _collect_results(
        self,
        result: ExecutionResult,
        exec_globals: dict,
        original_datasets: dict,
    ) -> ExecutionResult:
        """exec_globals içinden figure, result_df ve değişiklikleri topla."""

        # Tüm Plotly ve Matplotlib Figure nesnelerini topla (fig, fig1, fig2, ...)
        for var_name, var_value in exec_globals.items():
            if var_name.startswith("_"):
                continue  # private değişkenleri atla
            if var_value is not None:
                if isinstance(var_value, go.Figure):
                    result.figures.append(var_value)
                elif matplotlib and isinstance(var_value, matplotlib.figure.Figure):
                    result.figures.append(var_value)

        # Matplotlib pyplot çağrılarıyla oluşturulan aktif figürleri topla
        if plt is not None:
            try:
                for num in plt.get_fignums():
                    fig_obj = plt.figure(num)
                    if fig_obj not in result.figures and fig_obj.axes:
                        result.figures.append(fig_obj)
                plt.close("all")
            except Exception:
                pass

        # result_df kontrolü
        rdf = exec_globals.get("result_df")
        if rdf is not None and isinstance(rdf, pd.DataFrame):
            result.result_df = rdf

        # İlk DataFrame (df kısayolu) değişti mi?
        first_key = next(iter(original_datasets)) if original_datasets else None
        modified_df = exec_globals.get("df")
        if (
            modified_df is not None
            and isinstance(modified_df, pd.DataFrame)
            and first_key
        ):
            original_first = original_datasets[first_key]["df"]
            if self._df_is_modified(modified_df, original_first):
                result.modified_df = modified_df

        # Diğer datasetler değişti mi?
        datasets_copy = exec_globals.get("datasets", {})
        for key, entry in original_datasets.items():
            original_df = entry["df"]
            current_copy = datasets_copy.get(key)
            if (
                current_copy is not None
                and isinstance(current_copy, pd.DataFrame)
                and self._df_is_modified(current_copy, original_df)
            ):
                result.modified_datasets[key] = current_copy

        return result

    @staticmethod
    def _df_is_modified(current: pd.DataFrame, original: pd.DataFrame) -> bool:
        """
        DataFrame değişmiş mi? Önce hızlı id kontrolü, sonra detaylı.
        """
        # Hızlı yol: aynı nesne mi?
        if current is original:
            return False

        # Detaylı karşılaştırma
        try:
            return not current.equals(original)
        except Exception:
            # equals() başarısız olursa yapısal kontrol
            return (
                len(current) != len(original)
                or list(current.columns) != list(original.columns)
            )

    # ------------------------------------------------------------------
    # Timeout ile çalıştırma (process-based, thread yerine)
    # ------------------------------------------------------------------
    def _exec_with_timeout(self, code: str, exec_globals: dict) -> None:
        """
        Timeout ile kod çalıştır (thread tabanlı).

        Thread aynı bellek alanını paylaştığı için exec_globals'taki
        değişiklikler (fig, result_df) çalıştırma sonrası doğrudan erişilebilir.

        NOT: Python'da thread'ler zorla durdurulamaz. Timeout aşılırsa
        TimeoutError fırlatılır; thread arka planda çalışmaya devam edebilir.
        """
        exception_holder = [None]

        def _target():
            try:
                exec(code, exec_globals)
            except Exception as e:
                exception_holder[0] = e

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"Kod {self.timeout}s içinde tamamlanamadı"
            )

        if exception_holder[0] is not None:
            raise exception_holder[0]

    # ------------------------------------------------------------------
    # Güvenlik
    # ------------------------------------------------------------------
    def _check_security(self, code: str) -> None:
        """
        Kod güvenlik taraması (AST tabanlı).

        Raises:
            SecurityError: Güvensiz yapı tespit edilirse.
        """
        # AST parse et (syntax error da burada yakalanır)
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise SecurityError(f"Sözdizimi hatası: {exc}") from exc

        # Tehlikeli AST düğümlerini kontrol et
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # from os import ... gibi import'lar (safe_import ikincil savunma)
                module = node.module or ""
                base = module.split(".")[0]
                if base not in ALLOWED_BASE_MODULES:
                    raise SecurityError(
                        f"'from {module} import ...' güvenlik nedeniyle engellenmiştir."
                    )
                for alias in node.names:
                    if alias.name in ("os", "sys", "subprocess", "shutil", "socket"):
                        raise SecurityError(
                            f"'from {module} import {alias.name}' engellenmiştir."
                        )

            elif isinstance(node, ast.Call):
                # eval(), exec(), compile(), open() çağrıları
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "open"):
                        raise SecurityError(
                            f"'{node.func.id}()' çağrısı güvenlik nedeniyle engellenmiştir."
                        )

            elif isinstance(node, ast.Attribute):
                # __import__, __builtins__ erişimi
                if node.attr in ("__import__", "__builtins__", "__subclasses__"):
                    raise SecurityError(
                        f"'{node.attr}' özel attribute erişimi engellenmiştir."
                    )

        # String-tabanlı hızlı tarama (yorum/string literal temizlenerek)
        cleaned_code = self._strip_comments_and_strings(code)
        blocked_tokens = [
            "os.system", "os.popen", "os.remove", "os.unlink",
            "os.rmdir", "os.mkdir", "os.makedirs", "os.walk",
            "subprocess.call", "subprocess.run", "subprocess.Popen",
            "sys.exit", "sys._getframe",
        ]
        for token in blocked_tokens:
            if token in cleaned_code:
                raise SecurityError(
                    f"'{token}' kullanımı güvenlik nedeniyle engellenmiştir."
                )

    @staticmethod
    def _strip_comments_and_strings(code: str) -> str:
        """
        Kod içindeki yorum satırlarını ve string literal'ları kaldır.
        String-tabanlı güvenlik taraması için.
        """
        # Üçlü tırnak string'leri kaldır
        code = re.sub(r'"""[\s\S]*?"""', '""""""', code)
        code = re.sub(r"'''[\s\S]*?'''", "'''''", code)

        # Tek tırnak string'leri kaldır (basit yaklaşım)
        code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
        code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)

        # Yorum satırlarını kaldır
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

        return code

    # ------------------------------------------------------------------
    # Kod ön işleme
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_code(code: str) -> str:
        """
        LLM kodunu çalıştırmadan önce ön işle:
        - Herhangi bir Figure.show() çağrısını kaldır
          (Streamlit zaten st.plotly_chart ile gösteriyor)
        """
        code = re.sub(
            r'^([ \t]*)\w+\.show\([^)]*\)[ \t]*$',
            r'\1# .show() kaldırıldı — otomatik gösterim',
            code,
            flags=re.MULTILINE,
        )
        return code

    # ------------------------------------------------------------------
    # Hata formatlama
    # ------------------------------------------------------------------
    @staticmethod
    def _format_error(exc: Exception) -> str:
        """Exception'ı kullanıcı dostu formata çevir."""
        error_tb = traceback.format_exc()
        tb_lines = error_tb.strip().split("\n")
        short_tb = "\n".join(tb_lines[-4:])
        return f"❌ Hata: `{type(exc).__name__}: {exc}`\n```\n{short_tb}\n```"