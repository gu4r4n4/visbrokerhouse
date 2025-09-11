"""
Module: gpt_extractor
Purpose: File-by-file extraction of insurer offers using GPT-5 with strict JSON Schema.
Language policy: Prompts/instructions in EN; extracted data must remain in Latvian as in the PDF.
Security: No PDF text is logged by default. Hard size/time limits. Deterministic (temperature=0).
Integration step later: persist to Supabase and add legacy fallback gate.
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from jsonschema import validate as js_validate, Draft202012Validator
from pypdf import PdfReader
from openai import OpenAI

# =============================
# 1) JSON Schema (contract)
# =============================
INSURER_OFFER_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "InsurerOfferExtraction_v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["document_id", "insurer_code", "programs"],
    "properties": {
        "document_id": {"type": "string", "minLength": 1},
        "insurer_code": {"type": "string", "minLength": 1},
        "source_notes": {"type": "string"},
        "programs": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "program_code",
                    "program_type",
                    "base_sum_eur",
                    "premium_eur",
                    "features",
                ],
                "properties": {
                    "program_code": {"type": "string", "minLength": 1},
                    "program_type": {"type": "string", "enum": ["base", "additional"]},
                    "base_sum_eur": {"type": "number"},
                    "premium_eur": {"type": "number"},
                    "features": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["value"],
                            "properties": {
                                "value": {
                                    "oneOf": [
                                        {"type": "string", "enum": ["v", "-"]},
                                        {"type": "number"},
                                        {
                                            "type": "string",
                                            "pattern": r"^[0-9]+(\.[0-9]+)?\s?(EUR|€|eur)$",
                                        },
                                    ]
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "provenance": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "page": {"type": "integer", "minimum": 1},
                                        "bbox": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 4,
                                            "maxItems": 4,
                                        },
                                        "source_text": {"type": "string"},
                                        "table_hint": {
                                            "type": "string",
                                            "enum": [
                                                "PAMATPROGRAMMA",
                                                "PAPILDPROGRAMMAS",
                                                "auto",
                                            ],
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

# Pre-compile schema for speed & early error messages
_SCHEMA_VALIDATOR = Draft202012Validator(INSURER_OFFER_SCHEMA)

# =============================
# 2) Feature names list (for completeness)
# =============================
FEATURE_NAMES: List[str] = [
    "Programmas nosaukums",
    "Pakalpojuma apmaksas veids",
    "Programmas kods",
    "Apdrošinājuma summa pamatpolisei, EUR",
    "Pacientu iemaksa",
    "Maksas ģimenes ārsta mājas vizītes, limits EUR",
    "Maksas ārsta-specialista konsultācija, limits EUR",
    "Profesora, docenta, internista konsultācija, limits EUR",
    "Homeopāts",
    "Psihoterapeits",
    "Sporta ārsts",
    "ONLINE ārstu konsultācijas",
    "Laboratoriskie izmeklējumi",
    "Maksas diagnostika, piem., rentgens, elektrokradiogramma, USG, utml.",
    "Augsto tehnoloģiju izmeklējumi, piem., MRG, CT, limits (reižu skaits vai EUR)",
    "Obligātās veselības pārbaudes, limits EUR",
    "Ārstnieciskās manipulācijas",
    "Medicīniskās izziņas",
    "Fizikālā terapija",
    "Procedūras",
    "Vakcinācija, limits EUR",
    "Maksas grūtnieču aprūpe",
    "Maksas onkoloģiskā, hematoloģiskā ārstēšana",
    "Neatliekamā palīdzība valsts un privātā (limits privātai, EUR)",
    "Maksas stacionārie pakalpojumi, limits EUR",
    "Maksas stacionārā rehabilitācija, limits EUR",
    "Ambulatorā rehabilitācija",
    "Pamatpolises prēmija 1 darbiniekam, EUR",
    "Piemaksa par plastikāta kartēm, EUR",
    "Zobārstniecība ar 50% atlaidi (pamatpolise)",
    "Zobārstniecība ar 50% atlaidi, apdrošinājuma summa (pp)",
    "Vakcinācija pret ērčiem un gripu",
    "Ambulatorā rehabilitācija (pp)",
    "Medikamenti ar 50% atlaidi",
    "Sports",
    "Kritiskās saslimšanas",
    "Maksas stacionārie pakalpojumi, limits EUR (pp)",
]

# =============================
# 3) Prompts (EN instructions, Latvian data)
# =============================
SYSTEM_PROMPT = (
    "You are an expert at reading Latvian health-insurance PDFs.\n"
    "Return STRICTLY valid JSON that conforms to InsurerOfferExtraction_v1. No extra text.\n"
    "Rules:\n"
    "- Do not miss any program or any detail. Analyze all tables/sections.\n"
    "- Prioritize 'PAMATPROGRAMMA' as the base program. Treat 'PAPILDPROGRAMMAS' or headings containing '(pp)' or '(papildprogramma)' as additional programs.\n"
    "- 'premium_eur' for the base program MUST come from 'Prēmija vienai personai, EUR' in PAMATPROGRAMMA.\n"
    "- 'base_sum_eur' should be taken from or near 'Apdrošinājuma summa vienai personai'.\n"
    "- Normalize numbers: remove thousand separators/spaces; use a dot as decimal. 'base_sum_eur' and 'premium_eur' must be numbers.\n"
    "- Feature values allowed ONLY: 'v', '-', a bare number (e.g., 80), or '<number> EUR' (e.g., '70 EUR').\n"
    "- Handle broken headers like 'Prēmija vienai\\npersonai, EUR'.\n"
    "- Whenever possible provide 'provenance.page' (1-based), 'table_hint' (PAMATPROGRAMMA/PAPILDPROGRAMMAS/auto), and a short 'source_text'.\n"
    "- Keep names and values in Latvian exactly as in the PDF. Do not translate.\n"
    "- Include a concise coverage summary in 'warnings' describing what tables/sections were visited and any ambiguities.\n"
)

USER_PROMPT_TEMPLATE = (
    "DOCUMENT_ID: {document_id}\n"
    "INSURER_HINT: {insurer_hint}\n\n"
    "TASK:\n"
    "Extract ONE JSON payload per the schema InsurerOfferExtraction_v1. Base program = table 'PAMATPROGRAMMA'. Additional programs = 'PAPILDPROGRAMMAS' or headings with '(pp)/(papildprogramma)'.\n\n"
    "REQUIRED FIELDS:\n"
    "- insurer_code\n"
    "- program_code\n"
    "- base_sum_eur  ← from/near 'Apdrošinājuma summa vienai personai'\n"
    "- premium_eur   ← from 'Prēmija vienai personai, EUR' in PAMATPROGRAMMA\n\n"
    "FEATURES (values must be 'v' / '-' / number / '<number> EUR'):\n"
    + "\n".join(f"- {name}" for name in FEATURE_NAMES)
    + "\n\nPDF TEXT (per page, UTF-8):\n{pdf_text}\n"
)

# =============================
# 4) PDF text extraction (text PDFs only; OCR hook placeholder)
# =============================

def pdf_to_text_pages(pdf_bytes: bytes, max_pages: int = 30) -> List[str]:
    """Extract text page-by-page using pypdf. For scanned PDFs add OCR later."""
    pages: List[str] = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages[:max_pages]):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        # Normalize common header linebreaks that split words/columns
        text = text.replace("\u00A0", " ").replace("\r", "\n")
        pages.append(text)
    return pages

# =============================
# 5) OpenAI client call w/ JSON Schema enforcement
# =============================

@dataclass
class GPTConfig:
    model: str = os.getenv("GPT_EXTRACTION_MODEL", "gpt-5")
    timeout: int = int(os.getenv("GPT_TIMEOUT_SECONDS", "60"))
    max_retries: int = int(os.getenv("GPT_MAX_RETRIES", "2"))
    log_prompts: bool = os.getenv("GPT_LOG_PROMPTS", "false").lower() == "true"

_client: Optional[OpenAI] = None

def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

class ExtractionError(Exception):
    pass


def call_gpt_extractor(document_id: str, insurer_hint: str, pdf_pages: List[str], cfg: Optional[GPTConfig] = None) -> Dict[str, Any]:
    cfg = cfg or GPTConfig()
    user_text = USER_PROMPT_TEMPLATE.format(
        document_id=document_id,
        insurer_hint=insurer_hint or "",
        pdf_text="\n\n".join(
            f"===== Page {i+1} =====\n{p[:20000]}" for i, p in enumerate(pdf_pages)
        ),
    )

    if cfg.log_prompts:
        print("SYSTEM:\n", SYSTEM_PROMPT)
        print("USER:\n", user_text[:4000], "... [truncated]")

    # Compose response with JSON schema mode
    last_err: Optional[Exception] = None
    for attempt in range(cfg.max_retries + 1):
        try:
            client = _client_singleton()
            resp = client.responses.create(
                model=cfg.model,
                input=[
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "text", "text": user_text}]},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "InsurerOfferExtraction_v1",
                        "schema": INSURER_OFFER_SCHEMA,
                        "strict": True,
                    },
                },
                temperature=0,
            )
            payload = resp.output_parsed  # typed dict
            _SCHEMA_VALIDATOR.validate(payload)
            return payload
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise ExtractionError(f"GPT extraction failed: {e}") from e


# =============================
# 6) Public service function
# =============================

def extract_offer_from_pdf_bytes(pdf_bytes: bytes, document_id: str, insurer_hint: str = "") -> Dict[str, Any]:
    """End-to-end: PDF bytes -> pages -> GPT -> validated dict."""
    if len(pdf_bytes) > 15 * 1024 * 1024:  # 15MB guard
        raise ExtractionError("PDF too large (limit 15MB)")
    pages = pdf_to_text_pages(pdf_bytes)
    if not any(p.strip() for p in pages):
        # OCR hook could be placed here later
        raise ExtractionError("No extractable text; OCR not implemented")
    payload = call_gpt_extractor(document_id=document_id, insurer_hint=insurer_hint, pdf_pages=pages)

    # Safety: ensure feature coverage includes required constants
    # (Model may omit some; here we do not mutate, only warn to logs.)
    missing = [f for f in FEATURE_NAMES if f not in payload["programs"][0]["features"]]
    if missing and os.getenv("EXTRACTION_WARN_ON_MISSING", "true").lower() == "true":
        print(f"[WARN] Potentially missing feature keys in first program: {missing[:8]} ...")

    return payload

# =============================
# 7) FastAPI route (drop-in)
# =============================
try:
    from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
    from fastapi import status as http_status
    from starlette.responses import JSONResponse
except Exception:
    # Allow this module to be imported without FastAPI installed for testing
    APIRouter = None


def get_router() -> "APIRouter":
    if APIRouter is None:
        raise RuntimeError("FastAPI not installed; install fastapi and starlette")

    router = APIRouter(prefix="/api/extract", tags=["extraction"])

    @router.post("/pdf", response_class=JSONResponse)
    async def extract_pdf(
        file: UploadFile = File(...),
        insurer_hint: Optional[str] = None,
    ) -> Any:
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "PDF required")
        data = await file.read()
        try:
            payload = extract_offer_from_pdf_bytes(data, document_id=file.filename, insurer_hint=insurer_hint or "")
        except ExtractionError as e:
            raise HTTPException(http_status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
        except Exception as e:
            raise HTTPException(http_status.HTTP_500_INTERNAL_SERVER_ERROR, f"Unexpected error: {e}")
        return payload

    return router

# =============================
# 8) Usage note (wire-up)
# =============================
# In your FastAPI app factory:
# from gpt_extractor import get_router
# app.include_router(get_router())
