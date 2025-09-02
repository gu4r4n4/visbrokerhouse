# app/parsers/router.py
import os
from typing import Dict, Any
from .docx_parser import parse_docx
from .pdf_parser import parse_pdf

def parse_document(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return parse_docx(path)
    if ext == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")
