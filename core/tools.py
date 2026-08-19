"""
Tool Calling Specifications & Parsers — ReAct döngüsü için araç tanımları ve ayrıştırıcılar.

Desteklenen Araçlar:
1. execute_sql: Veritabanında SQL sorgusu çalıştırıp result_df döndürür.
2. inspect_csv_data: Belirli bir kolonun frekans dağılımını veya tekil değerlerini inceler.
3. read_csv_sample: CSV/SQLite tablosundan doğrudan örnek satırları okur.
4. generate_python_plot: result_df üzerinde Plotly görselleştirmesi (fig) üretir.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolCall:
    """Ayrıştırılmış araç çağrısı."""
    name: str
    arguments: Dict[str, Any]
    raw_text: Optional[str] = None
    call_id: Optional[str] = None

    def get_arg(self, *keys: str, default: Any = "") -> Any:
        """Birden çok olası anahtar adından ilk bulunanı döndür."""
        for k in keys:
            if k in self.arguments and self.arguments[k]:
                return self.arguments[k]
        return default


# ─────────────────────────────────────────────────────────
# OpenAI Uyumlu Araç Şemaları (Tools Definition)
# ─────────────────────────────────────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "SQLite veritabanında salt-okunur (SELECT/WITH) SQL sorgusu çalıştırır. "
                "Veri filtreleme, toplama (SUM, AVG, COUNT), gruplama (GROUP BY), sıralama (ORDER BY) "
                "ve tablo birleştirme (JOIN) işlemleri için bu aracı kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Çalıştırılacak standart SQLite SELECT sorgusu.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_csv_data",
            "description": (
                "CSV / SQLite tablosundaki bir kolonun değer dağılımını (frekans / value_counts) ve yüzdelerini hesaplar."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Hedef tablo adı.",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "İncelenecek kolon adı (örn: 'Marka', 'Kategori', 'Sehir').",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Kaç adet en sık geçen değer getirilsin (varsayılan: 10).",
                    },
                },
                "required": ["table_name", "column_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_python_plot",
            "description": (
                "SQL sorgusundan dönen temiz `result_df` DataFrame'i üzerinde Plotly ile interaktif grafik oluşturur. "
                "Grafik nesnesi `fig` değişkenine atanmalıdır. `pd`, `px`, `go` zaten tanımlıdır."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "result_df kullanarak fig üreten Plotly Python kodu.",
                    }
                },
                "required": ["python_code"],
            },
        },
    },
]


class ToolParser:
    """
    LLM yanıtlarından araç çağrılarını (Tool Calls) ve serbest düşünceyi (Thought / Explanation)
    yüksek toleransla ayrıştıran parser.
    """

    @classmethod
    def parse_response(cls, text: str) -> Tuple[str, List[ToolCall]]:
        """
        LLM yanıtını (Thought / Açıklama) ve (ToolCall Listesi) olarak ikiye ayırır.

        Desteklenen formatlar:
        1. Standart JSON Tool Calling blokları
        2. Markdown kod blokları: ```sql ... ``` ve ```python ... ```
        3. ReAct Tag blokları: <tool_call>...</tool_call>
        4. Action / Action Input formatı

        Returns:
            (thought_and_explanation, tool_calls_list)
        """
        if not text or not text.strip():
            return "", []

        tool_calls: List[ToolCall] = []

        # ── 1) JSON Tag / Block Parsing ──
        json_pattern = r"```(?:json)?\s*(\{\s*\"(?:tool|action|name)\"[\s\S]*?\})\s*```"
        json_matches = list(re.finditer(json_pattern, text, re.DOTALL | re.IGNORECASE))

        for match in json_matches:
            try:
                data = json.loads(match.group(1))
                tool_name = data.get("tool") or data.get("action") or data.get("name")
                args = data.get("arguments") or data.get("action_input") or data.get("parameters") or {}
                if isinstance(args, str):
                    if tool_name == "execute_sql":
                        args = {"query": args, "sql": args}
                    elif tool_name == "generate_python_plot":
                        args = {"python_code": args, "code": args}
                elif isinstance(args, dict):
                    # Anahtar eşitlemesi
                    if "query" in args:
                        args["sql"] = args["query"]
                    if "sql" in args:
                        args["query"] = args["sql"]
                    if "python_code" in args:
                        args["code"] = args["python_code"]
                    if "code" in args:
                        args["python_code"] = args["code"]

                if tool_name in ("execute_sql", "generate_python_plot", "inspect_csv_data", "read_csv_sample"):
                    tool_calls.append(ToolCall(name=tool_name, arguments=args, raw_text=match.group(0)))
            except Exception:
                pass

        # ── 2) Standart Markdown Kod Blokları (Doğrudan SQL ve Python) ──
        if not tool_calls:
            # SQL Blokları: ```sql SELECT ... ``` veya ```sqlite ... ``` veya ``` SELECT ... ```
            sql_pattern = r"```(?:sql|sqlite)?\s*([\s\S]*?)```"
            for match in re.finditer(sql_pattern, text, re.IGNORECASE):
                block_content = match.group(1).strip()
                if block_content.upper().startswith(("SELECT", "WITH", "EXPLAIN")):
                    tool_calls.append(
                        ToolCall(
                            name="execute_sql",
                            arguments={"query": block_content, "sql": block_content},
                            raw_text=match.group(0),
                        )
                    )

            # Python Plot Blokları: ```python fig = ... ``` veya ```py ... ```
            py_pattern = r"```(?:python|py)?\s*([\s\S]*?)```"
            for match in re.finditer(py_pattern, text, re.IGNORECASE):
                code = match.group(1).strip()
                if any(kw in code for kw in ("px.", "go.", "fig =", "fig.")):
                    tool_calls.append(
                        ToolCall(
                            name="generate_python_plot",
                            arguments={"python_code": code, "code": code},
                            raw_text=match.group(0),
                        )
                    )

        # ── 3) Çerçevesiz (Bare) SQL Sorguları (Son Çare) ──
        if not tool_calls:
            bare_sql_match = re.search(r"(?:^|\n)\s*((?:SELECT|WITH)\b[\s\S]+?;)", text, re.IGNORECASE)
            if bare_sql_match:
                bare_query = bare_sql_match.group(1).strip()
                tool_calls.append(
                    ToolCall(
                        name="execute_sql",
                        arguments={"query": bare_query, "sql": bare_query},
                        raw_text=bare_query,
                    )
                )

        # ── 4) Açıklama Metnini Temizleme ──
        clean_text = text
        for tc in tool_calls:
            if tc.raw_text:
                clean_text = clean_text.replace(tc.raw_text, "")

        clean_text = re.sub(r"\n{3,}", "\n\n", clean_text).strip()

        return clean_text, tool_calls
