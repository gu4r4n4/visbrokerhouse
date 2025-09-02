# app/parsers/pdf_parser.py
from typing import List, Dict, Any
import pdfplumber

def _normalize_table(tb: List[List[str]]) -> List[List[str]]:
    maxlen = max((len(r) for r in tb), default=0)
    norm = []
    for r in tb:
        r = [(c or "").strip() for c in (r or [])]
        if len(r) < maxlen:
            r += [""] * (maxlen - len(r))
        norm.append(r)
    return norm

def parse_pdf(path: str) -> Dict[str, Any]:
    texts: List[str] = []
    tables: List[List[List[str]]] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text(layout=True) or ""
            if txt:
                texts.append(txt)

            lattice = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            tbs = page.extract_tables(lattice) or []
            if not tbs:
                stream = {"vertical_strategy": "text", "horizontal_strategy": "text"}
                tbs = page.extract_tables(stream) or []
            for tb in tbs:
                if tb:
                    tables.append(_normalize_table(tb))
    return {"raw_text": "\n".join(texts).strip(), "tables": tables}
