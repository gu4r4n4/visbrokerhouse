import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.parsers.router import parse_document
from app.ai.extract import ai_enrich_and_validate

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Insurance Offer Extractor", version="0.1.0")

# Allow your UI to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    # For security, later replace "*" with your exact UI origin(s), e.g. "https://yourproject.lovable.dev"
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

# ---------- Models ----------
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
    offer_ids: List[int] = Field(default_factory=list)


# ---------- Helpers ----------
def _save_offers_to_db(
    programs: List[Program | Dict[str, Any]],
    source_file: str,
    company_hint: Optional[str],
    inquiry_id: Optional[int],
) -> List[int]:
    """
    Inserts parsed offers into public.offers and returns their IDs.
    If DB_URI is missing or insert fails, returns [] without failing the request.
    """
    uri = os.getenv("DB_URI")
    if not uri:
        # DB not configured: just skip writes
        return []

    try:
        import psycopg
        from psycopg.types.json import Json

        ids: List[int] = []
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                for p in programs:
                    data = p.model_dump() if hasattr(p, "model_dump") else dict(p)
                    cur.execute(
                        """
                        INSERT INTO public.offers
                          (insurer, company_hint, program_code, source, filename, inquiry_id,
                           base_sum_eur, premium_eur, payment_method, features, raw_json, status)
                        VALUES
                          (%s, %s, %s, %s, %s, %s,
                           %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            data.get("insurer"),
                            company_hint,
                            data.get("program_code"),
                            "api",
                            source_file,
                            inquiry_id,
                            data.get("base_sum_eur"),
                            data.get("premium_eur"),
                            data.get("payment_method"),
                            Json(data.get("features") or {}),
                            Json(data),
                            "parsed",
                        ),
                    )
                    ids.append(cur.fetchone()[0])
            conn.commit()
        return ids
    except Exception as e:
        # Don’t break the whole request on DB failure
        print("DB insert failed:", e)
        return []


# ---------- Routes ----------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


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

        # ensure response objects are Program instances
        progs: List[Program] = []
        for p in programs:
            if hasattr(p, "model_dump"):
                data = p.model_dump()
            elif isinstance(p, dict):
                data = p
            else:
                data = dict(p)
            progs.append(Program(**data))

        # write to DB (if configured)
        offer_ids = _save_offers_to_db(
            programs=progs,
            source_file=file.filename or safe_name,
            company_hint=company_hint,
            inquiry_id=inquiry_id,
        )

        # cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass

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
    """
    Connectivity check to Supabase Postgres.
    Returns server version and whether the 'offers' table exists.
    """
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
