"""
Data Lineage & XAI (Açıklanabilirlik) — Verinin ham halden nihai sayıya kadar geçirdiği dönüşüm adımlarını görselleştiren soy ağacı motoru.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LineageStep:
    step_type: str  # "SOURCE", "FILTER", "GROUPBY", "AGGREGATE", "SORT", "OUTPUT"
    title: str
    description: str
    icon: str


class DataLineageTracker:
    """
    Python/Pandas veya SQL kodunu analiz ederek veri dönüşüm adımlarını (Pipeline DAG)
    ve Mermaid.js akış diyagramını çıkaran Açıklanabilir Yapay Zeka (XAI) motoru.
    """

    @classmethod
    def trace_pipeline(
        cls,
        code: str,
        code_type: str,
        dataset_name: str,
        input_rows: int,
        output_rows: int,
    ) -> Tuple[List[LineageStep], str]:
        """
        Kodu analiz eder, adımları çıkarır ve Mermaid diyagramı döndürür.
        """
        steps: List[LineageStep] = []

        # 1. Kaynak Veri Adımı
        steps.append(
            LineageStep(
                step_type="SOURCE",
                title="1. Ham Veri Kaynağı",
                description=f"Dosya: `{dataset_name}` ({input_rows:,} satır)",
                icon="📂",
            )
        )

        code_lower = code.lower()

        # 2. Filtreleme / WHERE
        filter_desc = None
        if code_type == "python":
            # df[df['col'] > x] veya .query()
            filter_match = re.search(r"df\[\s*\(?df\[['\"]([^'\"]+)['\"]\]\s*([><=!]+)\s*([^\]\)]+)", code)
            if filter_match:
                filter_desc = f"Kolon: `{filter_match.group(1)}` {filter_match.group(2)} {filter_match.group(3).strip()}"
        else:
            where_match = re.search(r"\bWHERE\b\s+([\s\S]*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|;|$)", code, re.IGNORECASE)
            if where_match:
                filter_desc = where_match.group(1).strip()

        if filter_desc:
            steps.append(
                LineageStep(
                    step_type="FILTER",
                    title="2. Koşul & Filtreleme",
                    description=filter_desc[:60],
                    icon="🔍",
                )
            )

        # 3. Gruplama & Boyutlar
        groupby_desc = None
        if code_type == "python":
            gb_match = re.search(r"\.groupby\(\s*(\[[^\]]+\]|['\"][^'\"]+['\"])", code)
            if gb_match:
                groupby_desc = f"Boyut: {gb_match.group(1)}"
            elif "value_counts" in code:
                groupby_desc = "Frekans Sayımı (Value Counts)"
        else:
            gb_match = re.search(r"\bGROUP BY\b\s+([^;\n]+)", code, re.IGNORECASE)
            if gb_match:
                groupby_desc = f"Boyut: `{gb_match.group(1).strip()}`"

        if groupby_desc:
            steps.append(
                LineageStep(
                    step_type="GROUPBY",
                    title="3. Gruplama & Kırılım",
                    description=groupby_desc[:50],
                    icon="📊",
                )
            )

        # 4. Agregasyon (Toplama, Ortalama, Sayma)
        agg_desc = None
        if any(w in code_lower for w in (".sum(", "sum(")):
            agg_desc = "Toplam Değer Hesaplama (SUM)"
        elif any(w in code_lower for w in (".mean(", "avg(")):
            agg_desc = "Ortalama Hesaplama (AVG)"
        elif any(w in code_lower for w in (".count(", "count(")):
            agg_desc = "Kayıt Sayımı (COUNT)"

        if agg_desc:
            steps.append(
                LineageStep(
                    step_type="AGGREGATE",
                    title="4. Metrik & Agregasyon",
                    description=agg_desc,
                    icon="🧮",
                )
            )

        # 5. Sıralama ve Limit
        sort_desc = None
        if any(w in code_lower for w in ("sort_values", "order by")):
            limit_match = re.search(r"(?:head\(|limit\s+)(\d+)", code_lower)
            lim = f"Top {limit_match.group(1)}" if limit_match else "Sıralama"
            sort_desc = f"{lim} (Azalan / Artan)"
            steps.append(
                LineageStep(
                    step_type="SORT",
                    title="5. Sıralama & Kısıtlama",
                    description=sort_desc,
                    icon="⚡",
                )
            )

        # 6. Nihai Çıktı Tablosu
        steps.append(
            LineageStep(
                step_type="OUTPUT",
                title="Nihai Tablo & Rapor",
                description=f"Sonuç: `{output_rows:,} satır` doğrulanmış veri",
                icon="🎯",
            )
        )

        # Mermaid Diyagramı Üretimi
        mermaid_lines = ["graph LR", "    classDef default fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;"]
        for i in range(len(steps) - 1):
            curr_id = f"N{i}"
            next_id = f"N{i+1}"
            curr_label = f'"{steps[i].icon} {steps[i].title}<br/><small>{steps[i].description}</small>"'
            next_label = f'"{steps[i+1].icon} {steps[i+1].title}<br/><small>{steps[i+1].description}</small>"'
            mermaid_lines.append(f"    {curr_id}[{curr_label}] --> {next_id}[{next_label}]")

        mermaid_str = "\n".join(mermaid_lines)
        return steps, mermaid_str
