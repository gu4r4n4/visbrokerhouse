import traceback
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from app.parsers.router import parse_document
from app.ai.extract import ai_enrich_and_validate

app = FastAPI(title="Insurance Offer Extractor", version="0.1.0")

class Program(BaseModel):
    # Core fields
    insurer: str = Field(..., description="Apdrošinātājs")
    program_code: str = Field(..., description="Programmas kods")
    base_sum_eur: float = Field(..., description="Apdrošinājuma summa pamatpolisei, EUR")
    premium_eur: float = Field(..., description="Pamatpolises prēmija 1 darbiniekam, EUR")
    payment_method: Optional[str] = Field(None, description="Pakalpojuma apmaksas veids")
    # Full 25-row table as LV keys -> value (string/number)
    features: Dict[str, Any] = Field(default_factory=dict)

class ExtractionResult(BaseModel):
    source_file: str
    programs: List[Program] = Field(default_factory=list)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/ingest", response_model=ExtractionResult)
async def ingest(file: UploadFile = File(...), company_hint: Optional[str] = Form(None)):
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
        progs = []
        for p in programs:
            if hasattr(p, "model_dump"):
                data = p.model_dump()
            elif isinstance(p, dict):
                data = p
            else:
                data = dict(p)
            progs.append(Program(**data))

        return ExtractionResult(source_file=file.filename, programs=progs)


    except Exception as e:
        # return error details so we can see the cause
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()},
        )



from fastapi.responses import RedirectResponse

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
                cur.execute("""
                    select exists (
                        select 1
                        from information_schema.tables
                        where table_schema = 'public'
                          and table_name   = 'offers'
                    )
                """)
                has_offers = cur.fetchone()[0]
        return {"ok": True, "version": version, "offers_table": has_offers}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
