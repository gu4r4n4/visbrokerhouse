from typing import Dict, Any, List
import pandas as pd

def parse_xlsx(path: str) -> Dict[str, Any]:
    xl = pd.ExcelFile(path)
    tables_serialized: List[List[List[str]]] = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype=str).fillna("")
        # serialize DataFrame to rows of strings
        rows = [df.columns.tolist()] + df.values.tolist()
        rows = [[str(c) for c in row] for row in rows]
        tables_serialized.append(rows)

    return {
        "filetype": "xlsx",
        "pages_text": [],
        "raw_text": "",  # optional
        "tables": tables_serialized,
    }
