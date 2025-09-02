import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.parsers.router import parse_document
from app.ai.extract import ai_enrich_and_validate

app = FastAPI(title="Insurance Offer Extractor", version="0.1.0")


class Program(BaseModel):
    insurer: str = Field(..., description="Apdrošinātājs")
    program_code: str = Field(..., description="Programmas kods")
    base_sum_eur: float = Field(..., description="Apdrošinājuma summa pamatpolisei, EUR")
    premium_eur: float = Field(..., description="Pamatpolises prēmija 1 darbiniekam, EUR")
    payment_method: Optional[str] = Field(None, description="Pakalpojuma apmaksas veids")
    features: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    source_file: str
    programs: List[Program] = Field(default_factory=list)
    inquiry_id: Optional[int] = None
    offer_ids: List[int] = Field(default_factory=list)  # DB ids of inserted rows


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


def _insert_offers(
    programs: List[Program],
    company_hint: Optional[str],
    filename: Optional[str],
    inquiry_id: Optional[int],
) -> List[int]:
    """Insert each program into public.offers. Returns list of new IDs.
       Best-effort: if DB_URI is missing or insert fails, we swallow errors and return []."""
    dsn = os.getenv("DB_URI")
    if not dsn:
        return []

    try:
        # local imports to avoid import-time issues
        import psycopg
        from psycopg.types.json import Jsonb

        ids: List[int] = []
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for p in programs:
                    cur.execute(
                        """
                        INSERT INTO public.offers
                          (insurer, company_hint, program_code, source, filename, inquiry_id,
                           base_sum_eur, premium_eur, payment_method, features, raw_json, status)
                        VALUES
                          (%s, %s, %s, 'upload', %s, %s,
                           %s, %s, %s, %s, %s, 'parsed')
                        RETURNING id
                        """,
                        (
                            p.insurer,
                            company_hint,
                            p.program_code,
                            filename,
                            inquiry_id,
                            p.base_sum_eur,
                            p.premium_eur,
                            p.payment_method,
                            Jsonb(p.features),
                            Jsonb(p.model_dump()),
                        ),
                    )
                    ids.append(cur.fetchone()[0])
        return ids
    except Exception:
        # don’t block user on DB errors
        return []


@app.post("/ingest", response_model=ExtractionResult)
async def ingest(
    file: UploadFile = File(...),
    company_hint: Optional[str] = Form(None),
    inquiry_id: Optional[int] = Form(None),
):
    try:
        # save to tmp (Windows-safe)
        tempdir = tempfile.gettempdir()
        safe_name = (file.filename or "offer.bin").replace("\\", "_").replace("/", "_")
        tmp_path = os.path.join(tempdir, safe_name)
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        # parse
        parsed = parse_document(tmp_path)

        # AI enrichment & validation to LV schema
        programs = await ai_enrich_and_validate(parsed, company_hint=company_hint)

        # clean up
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        # ensure response is Program (not ProgramModel)
        progs: List[Program] = []
        for p in programs:
            if hasattr(p, "model_dump"):
                data = p.model_dump()
            elif isinstance(p, dict):
                data = p
            else:
                data = dict(p)
            progs.append(Program(**data))

        # write to DB (best-effort)
        offer_ids = _insert_offers(
            programs=progs,
            company_hint=company_hint,
            filename=file.filename,
            inquiry_id=inquiry_id,
        )

        return ExtractionResult(
            source_file=file.filename,
            programs=progs,
            inquiry_id=inquiry_id,
            offer_ids=offer_ids,
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/db/ping")
def db_ping():
    """Connectivity check to Supabase Postgres; verifies offers table exists."""
    import psycopg  # local import

    uri = os.getenv("DB_URI")
    if not uri:
        raise HTTPException(status_code=500, detail="DB_URI not set")

    try:
        with psycopg.connect(uri, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("select version()")
                version = cur.fetchone()[0]
                cur.execute(
                    """
                    select exists (
                        select 1
                        from information_schema.tables
                        where table_schema = 'public'
                          and table_name   = 'offers'
                    )
                    """
                )
                has_offers = cur.fetchone()[0]
        return {"ok": True, "version": version, "offers_table": has_offers}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
