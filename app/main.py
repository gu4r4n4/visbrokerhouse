import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional
from app.ai.extract import ai_enrich_and_validate

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.parsers.router import parse_document
from app.ai.extract import ai_enrich_and_validate

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Insurance Offer Extractor", version="0.1.0")

# Allow your UI to call the API from the browser
app.add_middleware(
    CORSMiddleware,
    # TODO: lock down to ["https://lovable.dev"] when ready
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS", "PATCH"],
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


class InquiryMeta(BaseModel):
    company_name: Optional[str] = None
    employees_count: Optional[int] = None


# ----- Share link models (no inquiry_id) -----
class ShareResult(BaseModel):
    source_file: str
    programs: List[Program]


class ShareSnapshotIn(BaseModel):
    """
    Snapshot payload coming from the UI to be shared publicly.
    """
    company_name: Optional[str] = None
    employees_count: Optional[int] = None
    results: List[ShareResult] = Field(default_factory=list)
    # default 30 days; None => no expiry
    expires_in_hours: Optional[int] = 720


class ShareTokenOut(BaseModel):
    token: str


class ShareViewOut(BaseModel):
    company_name: Optional[str] = None
    employees_count: Optional[int] = None
    results: List[ShareResult] = Field(default_factory=list)


# ---------- Helpers ----------
def _maybe_update_inquiry_meta(
    inquiry_id: Optional[int],
    *,
    company_name: Optional[str],
    employees_count: Optional[int],
) -> None:
    """
    If inquiry_id and at least one field provided, update
    public.insurance_inquiries(company_name, employees_count).
    Silently no-ops if DB_URI is missing.
    """
    if not inquiry_id:
        return
    if company_name is None and employees_count is None:
        return

    uri = os.getenv("DB_URI")
    if not uri:
        return

    try:
        import psycopg
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.insurance_inquiries
                    SET
                      company_name    = COALESCE(%s, company_name),
                      employees_count = COALESCE(%s, employees_count)
                    WHERE id = %s
                    """,
                    (company_name, employees_count, inquiry_id),
                )
            conn.commit()
    except Exception as e:
        # Don't break the request just because meta update failed
        print("Inquiry meta update failed:", e)


def _save_offers_to_db(
    programs: List[Program | Dict[str, Any]],
    source_file: str,
    company_hint: Optional[str],
    inquiry_id: Optional[int],
) -> List[int]:
    """
    Inserts parsed offers into public.offers and returns their IDs.
    If inquiry_id is missing/invalid, falls back to NULL (so offers are still saved).
    If DB_URI is missing or insert fails completely, returns [] without failing the request.
    """
    uri = os.getenv("DB_URI")
    if not uri:
        return []

    try:
        import psycopg
        from psycopg.types.json import Json

        ids: List[int] = []
        with psycopg.connect(uri) as conn:
            with conn.cursor() as cur:
                for p in programs:
                    data = p.model_dump() if hasattr(p, "model_dump") else dict(p)
                    try:
                        # First try with provided inquiry_id (may be None)
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
                                inquiry_id,  # may be None; FK only checked when not null
                                data.get("base_sum_eur"),
                                data.get("premium_eur"),
                                data.get("payment_method"),
                                Json(data.get("features") or {}),
                                Json(data),
                                "parsed",
                            ),
                        )
                        ids.append(cur.fetchone()[0])

                    except psycopg.errors.ForeignKeyViolation:
                        # Bad inquiry_id → retry once with NULL
                        cur.execute(
                            """
                            INSERT INTO public.offers
                              (insurer, company_hint, program_code, source, filename, inquiry_id,
                               base_sum_eur, premium_eur, payment_method, features, raw_json, status)
                            VALUES
                              (%s, %s, %s, %s, %s, NULL,
                               %s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                data.get("insurer"),
                                company_hint,
                                data.get("program_code"),
                                "api",
                                source_file,
                                data.get("base_sum_eur"),
                                data.get("premium_eur"),
                                data.get("payment_method"),
                                Json(data.get("features") or {}),
                                Json(data),
                                "parsed",
                            ),
                        )
                        ids.append(cur.fetchone()[0])

                    except Exception as e:
                        # Log and continue with remaining programs
                        print("DB insert for one program failed:", e)

            conn.commit()
        return ids

    except Exception as e:
        print("DB insert failed (connection-level):", e)
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
    # Extra form fields coming from UI (saved to inquiry meta, not required in response)
    company_name: Optional[str] = Form(None),
    employee_count: Optional[int] = Form(None),
):
    """
    Accepts a PDF + metadata, parses & enriches programs, returns JSON and saves to DB.
    NOTE: send employee_count as a NUMBER (e.g., 45), not text like "~ 45 cilvēki".
    """
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

        # best-effort: update inquiry meta if provided
        _maybe_update_inquiry_meta(
            inquiry_id,
            company_name=company_name,
            employees_count=employee_count,
        )

        # write offers to DB (if configured)
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


# Accept both POST and PATCH for UI compatibility
@app.post("/inquiries/{inquiry_id}/meta")
@app.patch("/inquiries/{inquiry_id}/meta")
def update_inquiry_metadata(inquiry_id: int, payload: InquiryMeta):
    """
    Optional endpoint for the UI to explicitly save inquiry meta.
    Succeeds even if only one field is provided. 404/400 guarded below.
    """
    if payload.company_name is None and payload.employees_count is None:
        raise HTTPException(status_code=400, detail="No fields to update")

    _maybe_update_inquiry_meta(
        inquiry_id,
        company_name=payload.company_name,
        employees_count=payload.employees_count,
    )
    return {"ok": True}


# -------- Share links WITHOUT inquiry_id (snapshot storage) --------
@app.post("/shares", response_model=ShareTokenOut)
def create_share_token(payload: ShareSnapshotIn):
    """
    Store a snapshot payload (company meta + results) under a random token.
    Returns { token } that can be used at GET /shares/{token}.
    """
    uri = os.getenv("DB_URI")
    if not uri:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DB unavailable",
        )

    import secrets
    from datetime import datetime, timedelta
    import psycopg
    from psycopg.types.json import Json

    token = secrets.token_urlsafe(24)
    expires_at = None
    if payload.expires_in_hours and payload.expires_in_hours > 0:
        expires_at = datetime.utcnow() + timedelta(hours=payload.expires_in_hours)

    snapshot = {
        "company_name": payload.company_name,
        "employees_count": payload.employees_count,
        "results": [r.model_dump() for r in payload.results],
    }

    try:
        with psycopg.connect(uri, autocommit=True) as conn:
            with conn.cursor() as cur:
                # Requires table: public.share_links(token text unique, payload jsonb, created_at timestamptz default now(), expires_at timestamptz null)
                cur.execute(
                    """
                    insert into public.share_links (token, payload, expires_at)
                    values (%s, %s, %s)
                    returning token
                    """,
                    (token, Json(snapshot), expires_at),
                )
                row = cur.fetchone()
                return {"token": row[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create share token: {e}")


@app.get("/shares/{token}", response_model=ShareViewOut)
def resolve_share_token(token: str):
    """
    Resolve a token and return the stored snapshot.
    404 if token not found or expired.
    """
    uri = os.getenv("DB_URI")
    if not uri:
        raise HTTPException(status_code=503, detail="DB unavailable")

    import psycopg
    from datetime import datetime, timezone

    with psycopg.connect(uri) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select payload, expires_at from public.share_links where token = %s",
                (token,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Share link not found")

            payload, expires_at = row

            if expires_at is not None:
                now = datetime.now(timezone.utc)
                if getattr(expires_at, "tzinfo", None) is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if now > expires_at:
                    raise HTTPException(status_code=404, detail="Share link expired")

    # payload is JSON with company_name, employees_count, results[]
    try:
        company_name = payload.get("company_name")
        employees_count = payload.get("employees_count")
        results_in = payload.get("results") or []
        # Validate/normalize into ShareResult -> Program models
        results_out: List[ShareResult] = []
        for r in results_in:
            progs = [Program(**p) for p in (r.get("programs") or [])]
            results_out.append(ShareResult(source_file=r.get("source_file") or "unknown", programs=progs))
        return ShareViewOut(
            company_name=company_name,
            employees_count=employees_count,
            results=results_out,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corrupt snapshot payload: {e}")


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
