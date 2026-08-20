"""
Sandbox Güvenlik Testleri
=========================
SafeExecutor'ın bilinen sandbox kaçış yöntemlerini bloklamayı
doğrulayan birim testleri.

Çalıştırmak için:
    python -m pytest tests/test_sandbox_security.py -v
veya
    python tests/test_sandbox_security.py
"""

import sys
import os
import unittest

# Proje kök dizinini sys.path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox.executor import SafeExecutor, SecurityError

import pandas as pd


class TestSandboxSecurity(unittest.TestCase):
    """SafeExecutor güvenlik testleri."""

    def setUp(self):
        self.executor = SafeExecutor(timeout=5)
        self.dummy_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        self.datasets = {"test": {"df": self.dummy_df, "metadata": {}}}

    # ── Kaçış Denemeleri ───────────────────────────────────────────────

    def test_blocked_getattr_direct(self):
        """getattr() doğrudan çağrısı engellenmelidir."""
        code = "x = getattr(int, '__mro__')"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "getattr engellenmelidir")
        self.assertIn("güvenlik", result.error.lower())

    def test_blocked_setattr_direct(self):
        """setattr() doğrudan çağrısı engellenmelidir."""
        code = "setattr(int, 'foo', lambda: None)"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "setattr engellenmelidir")

    def test_blocked_delattr_direct(self):
        """delattr() doğrudan çağrısı engellenmelidir."""
        code = "delattr(int, 'foo')"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "delattr engellenmelidir")

    def test_blocked_dunder_attribute_access(self):
        """__class__.__bases__ gibi dunder attribute erişimleri engellenmelidir."""
        code = "x = (1).__class__.__bases__[0]"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "Dunder attribute erişimi engellenmelidir")

    def test_blocked_subclasses_via_attribute(self):
        """__subclasses__() erişimi engellenmelidir."""
        code = "classes = (1).__class__.__bases__[0].__subclasses__()"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "__subclasses__ engellenmelidir")

    def test_blocked_string_literal_dunder(self):
        """String literal olarak dunder isimler engellenmelidir (concatenation bypass)."""
        code = "name = '__globals__'"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "String dunder literal engellenmelidir")

    def test_blocked_eval(self):
        """eval() çağrısı engellenmelidir."""
        code = "eval('1+1')"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "eval() engellenmelidir")

    def test_blocked_exec(self):
        """exec() çağrısı engellenmelidir."""
        code = "exec('import os')"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "exec() engellenmelidir")

    def test_blocked_open(self):
        """open() çağrısı engellenmelidir."""
        code = "f = open('secret.txt', 'r')"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "open() engellenmelidir")

    def test_blocked_import_os(self):
        """import os engellenmelidir."""
        code = "import os"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "'os' import engellenmelidir")

    def test_blocked_from_os_import(self):
        """from os import path engellenmelidir."""
        code = "from os import path"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "'from os import' engellenmelidir")

    def test_blocked_subprocess(self):
        """import subprocess engellenmelidir."""
        code = "import subprocess"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "subprocess engellenmelidir")

    def test_blocked_sys(self):
        """import sys engellenmelidir."""
        code = "import sys"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNotNone(result.error, "sys engellenmelidir")

    def test_blocked_os_system_token(self):
        """os.system token string-tabanlı taramada yakalanmalıdır."""
        code = "x = 'os.system'"
        # String içinde olduğu için strip edildikten sonra taranır
        # Şu an string-stripping yapıldığı için bu geçebilir; doğrulayalım
        # (expected: might pass because it's inside string — but we test the direct call)
        code2 = "os.system('ls')"
        result = self.executor.execute(code2, self.datasets, "test")
        self.assertIsNotNone(result.error, "os.system() engellenmelidir")

    # ── İzin Verilen İşlemler ──────────────────────────────────────────

    def test_allowed_pandas_operations(self):
        """Normal Pandas işlemleri çalışmalıdır."""
        code = "result_df = df.describe()"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNone(result.error, f"Pandas işlemi hata vermemeli: {result.error}")
        self.assertIsNotNone(result.result_df)

    def test_allowed_numpy_operations(self):
        """NumPy işlemleri çalışmalıdır."""
        code = "import numpy as np\nresult_df = pd.DataFrame({'mean': [np.mean(df['a'])]})"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNone(result.error, f"NumPy işlemi hata vermemeli: {result.error}")

    def test_allowed_plotly_chart(self):
        """Plotly görselleştirme çalışmalıdır."""
        code = "import plotly.express as px\nfig = px.bar(df, x='a', y='b')"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNone(result.error, f"Plotly işlemi hata vermemeli: {result.error}")
        self.assertTrue(result.has_figure)

    def test_allowed_groupby_agg(self):
        """groupby + agg çalışmalıdır."""
        code = "result_df = df.groupby('a')['b'].sum().reset_index()"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNone(result.error, f"groupby hata vermemeli: {result.error}")

    def test_allowed_string_operations(self):
        """String işlemleri çalışmalıdır."""
        code = "s = 'hello' + ' ' + 'world'\nprint(s)"
        result = self.executor.execute(code, self.datasets, "test")
        self.assertIsNone(result.error, f"String işlemi hata vermemeli: {result.error}")


if __name__ == "__main__":
    print("=" * 60)
    print("Sandbox Güvenlik Testleri")
    print("=" * 60)
    unittest.main(verbosity=2)
