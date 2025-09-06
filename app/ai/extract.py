# app/ai/extract.py
import os, json, re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .schema import json_schema, LV_FEATURE_KEYS
from .prompts import SYSTEM, build_prompt

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ----------------- Normalization helpers -----------------
KEY_ALIASES = {
    "Maksas grūtniecību aprūpe": "Maksas grūtnieču aprūpe",
    "Maksas grūtniecu aprūpe": "Maksas grūtnieču aprūpe",
    "Arstnieciskās manipulācijas": "Ārstnieciskās manipulācijas",
}
# NOTE: also strip SOFT HYPHEN U+00AD which often appears in wrapped PDF cells
_CTRL_RE = re.compile(r"[\x00-\x1F\x7F\u00AD]")

def _clean_key(k: str) -> str:
    k = _CTRL_RE.sub("", k or "").replace("\u00A0", " ").strip()
    k = k.replace("Arstnieciskās", "Ārstnieciskās")
    return KEY_ALIASES.get(k, k)

def normalize_features(raw: dict, insurer: str, program_code: str, premium_eur: float) -> dict:
    norm = {k: "-" for k in LV_FEATURE_KEYS}
    for k, v in (raw or {}).items():
        ck = _clean_key(str(k))
        if ck in norm:
            norm[ck] = v
    if "Apdrošinātājs" in norm:
        norm["Apdrošinātājs"] = insurer
    if "Programmas kods" in norm:
        norm["Programmas kods"] = program_code  # keep exactly as displayed (incl. '+')
    if "Pamatpolises prēmija 1 darbiniekam, EUR" in norm and premium_eur is not None:
        norm["Pamatpolises prēmija 1 darbiniekam, EUR"] = premium_eur
    if "Pamatpolises prēmija 1 darbiniekam" in norm:
        norm["Pamatpolises prēmija 1 darbiniekam"] = premium_eur
    return norm

# ----------------- STRICT company_hint → code mapping -----------------
def _company_hint_to_code(hint: Optional[str]) -> Optional[str]:
    """
    Map company_hint to the strict codes required by the UI.
    - BTA/BTA2 normalize to brand 'BTA' (special logic stays elsewhere).
    - Other supported insurers map to EXACT codes below (and nothing else).
    - If no match, return the original hint unchanged (do NOT invent codes).
    """
    if not hint:
        return None
    raw = hint
    h = raw.strip().lower()
    if h.startswith("bta"):
        return "BTA"
    mapping = {
        "compensa": "COM_VA",
        "seesam": "SEE_VA",
        "balta": "BAL_VA",
        "ban": "BAN_VA",
        "ergo": "ERG_VA",
        "gjensidige": "GJE_VA",
        "if": "IFI_VA",
    }
    return mapping.get(h, raw)

# ----------------- DTOs -----------------
class ProgramModel(BaseModel):
    insurer: str
    program_code: str
    base_sum_eur: float
    premium_eur: float
    payment_method: str | None = None
    features: dict

class ExtractionModel(BaseModel):
    programs: List[ProgramModel] = Field(default_factory=list)

# ----------------- Utils -----------------
def _norm_text(s: Optional[str]) -> str:
    if s is None: return ""
    # strip control chars + soft hyphen, normalize NBSP → space
    return _CTRL_RE.sub("", str(s)).replace("\u00A0", " ").strip()

# Prefer matching a code with '+' (e.g., V2+) and fall back to V2
def _find_prog_token(s: str) -> Optional[str]:
    t = _norm_text(s)
    # 1) Prefer the variant with '+'
    m = re.search(r"(?<!\w)V\s*\d+\s*\+(?!\w)", t, re.IGNORECASE)
    if m:
        return re.sub(r"\s+", "", m.group(0))
    # 2) Plain V\d
    m = re.search(r"(?<!\w)V\s*\d+(?!\w)", t, re.IGNORECASE)
    if m:
        return re.sub(r"\s+", "", m.group(0))
    # 3) Pamatprogramma
    if re.search(r"\bPamatprogramma\b", t, re.IGNORECASE):
        return "Pamatprogramma"
    return None

def _same_prog(a: str, b: str) -> bool:
    return re.sub(r"\s+", "", (a or "")).lower() == re.sub(r"\s+", "", (b or "")).lower()

# numbers helper: returns floats found, ignoring those immediately followed by '%'
_NUM_TOKEN_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[ \u00A0]\d{3})+|\d+)(?:[.,]\d+)?(?!\d)")
def _extract_numbers_no_percent(s: str) -> List[float]:
    if not s: return []
    t = _norm_text(s).replace("€","")
    out = []
    for m in _NUM_TOKEN_RE.finditer(t):
        span = m.span()
        # look ahead a couple chars to exclude percents
        tail = t[span[1]:span[1]+2]
        if "%" in tail:
            continue
        token = m.group(0).replace(" ", "").replace("\u00A0","").replace(",", ".")
        try:
            out.append(float(token))
        except:
            pass
    return out

def _num_safe(s: str) -> Optional[float]:
    nums = _extract_numbers_no_percent(s)
    return nums[0] if nums else None

def _find_col_idx(headers: List[str], candidates: List[str]) -> Optional[int]:
    hdrs = [(_norm_text(h)).lower() for h in headers]
    for i, h in enumerate(hdrs):
        for cand in candidates:
            if cand.lower() in h:
                return i
    return None

# ----------------- BTA2 singleton helpers (no filename reliance) -----------------
def _first_program_anchor_from_text(raw_text: str) -> Optional[str]:
    """
    Returns the FIRST program anchor seen in text.
    Prefers V# or V#+ to Pamatprogramma when both show up in the same span.
    """
    if not raw_text:
        return None
    # scan with a small window so we can prefer V# over Pamatprogramma if adjacent
    for m in re.finditer(r"(V\d+\+?|Pamatprogramma)", raw_text, re.IGNORECASE):
        token = m.group(1)
        if re.match(r"(?i)V\d+\+?$", token):
            return token.replace(" ", "")
        # otherwise remember pamat, but keep scanning a few steps ahead for V#
        ahead = raw_text[m.start(): m.start() + 150]
        mv = re.search(r"(?i)V\s*\d+\+?", ahead)
        if mv:
            return re.sub(r"\s+", "", mv.group(0))
        return "Pamatprogramma"
    return None

def _force_single_for_bta2(programs: List[ProgramModel], parsed: Dict[str, Any]) -> List[ProgramModel]:
    """
    Force exactly ONE program for BTA2:
      1) If we can detect a first program anchor in raw_text -> keep that match.
      2) Else prefer any V# or V#+ program over Pamatprogramma.
      3) Else keep the first item.
    """
    if not programs:
        return programs
    raw = parsed.get("raw_text") or ""

    anchor = _first_program_anchor_from_text(raw)
    if anchor:
        # exact match by program_code or Programmas nosaukums
        exact = [p for p in programs if _same_prog(p.program_code, anchor) or _same_prog(p.features.get("Programmas nosaukums", ""), anchor)]
        if exact:
            # choose the most complete row (larger base_sum then premium)
            best = max(exact, key=lambda p: (p.base_sum_eur or 0, p.premium_eur or 0))
            return [best]

    # prefer V# / V#+ over Pamatprogramma
    v_like = [p for p in programs if re.match(r"(?i)^\s*v\d+\+?\s*$", p.program_code) or re.match(r"(?i)^\s*v\d+\+?\s*$", str(p.features.get("Programmas nosaukums", "")))]
    if v_like:
        best = max(v_like, key=lambda p: (p.base_sum_eur or 0, p.premium_eur or 0))
        return [best]

    # otherwise just keep the first (stable)
    return [programs[0]]

# ----------------- Scored header finder -----------------
def _find_program_table_layout_scored(tbl: List[List[str]]) -> tuple[Optional[Dict[str, int]], int]:
    """
    Detect the header row + correct columns inside ONE table for the MAIN program grid.
    Among multiple 'Prēmija 1 personai (EUR)' columns, prefer the one that:
      - has many valid premiums (50..5000) on rows that look like V2+/V3+/Pamatprogramma
      - has larger typical premiums (avg/max), and
      - whose header hints 'Visiem' or '(52)'.

    Returns: (layout_dict or None, score)
    layout_dict keys: header_row, c_program, c_sum, c_prem
    """
    import unicodedata

    def _canon(s: str) -> str:
        t = _norm_text(s)
        t = unicodedata.normalize("NFD", t)
        t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
        return t.lower()

    best: Optional[Dict[str, int]] = None
    best_score = 0

    for r_idx, row in enumerate(tbl or []):
        row_norm = [_norm_text(c) for c in row]
        if not any(row_norm):
            continue

        # ---- build STACKED headers for every column across ±2 rows ----
        top = max(0, r_idx - 2)
        bot = min(len(tbl) - 1, r_idx + 2)
        max_cols = max((len(r) for r in tbl[top:bot + 1] if r), default=len(row_norm))

        def stacked_header(col: int) -> str:
            parts: List[str] = []
            for rr in range(top, bot + 1):
                if col < len(tbl[rr]):
                    parts.append(_norm_text(tbl[rr][col]))
            return _canon(" ".join(parts))

        # ---- detect columns by STACKED headers (not single-row) ----
        c_program = None
        c_sum = None
        prem_idxs: List[int] = []

        for c in range(max_cols):
            h = stacked_header(c)
            if c_program is None and ("program" in h or "programma" in h):
                c_program = c
            if c_sum is None and ("apdrosinajuma" in h and "summa" in h):
                c_sum = c
            if (("premij" in h or "prem" in h) and ("person" in h or "eur" in h)):
                prem_idxs.append(c)

        if c_program is None or c_sum is None or not prem_idxs:
            continue

        # ---- score each candidate premium column ----
        for c_prem in prem_idxs:
            header = stacked_header(c_prem)

            # strong bonus for the "everyone (52)" column
            header_bonus = 0
            if "visiem" in header:
                header_bonus += 120
            if re.search(r"\(\s*52\s*\)", header) or re.search(r"(^|\D)52(\D|$)", header):
                header_bonus += 120

            premiums: List[float] = []
            valid_cnt = 0
            bigsum_cnt = 0

            # inspect next ~20 rows; detect program rows robustly
            for rr in range(r_idx + 1, min(len(tbl), r_idx + 20)):
                r = tbl[rr]
                rtxt = " ".join(_norm_text(x) for x in r)
                looks_like_program = bool(_find_prog_token(rtxt))
                if not looks_like_program:
                    continue

                base_sum = _num_safe(r[c_sum])  if c_sum  < len(r) else None
                prem_val = _num_safe(r[c_prem]) if c_prem < len(r) else None

                if base_sum and base_sum >= 1000:
                    bigsum_cnt += 1
                if prem_val and 50 <= prem_val <= 5000:
                    premiums.append(prem_val)
                    valid_cnt += 1

            if not premiums:
                score = 0
            else:
                premiums.sort()
                median = premiums[len(premiums)//2]
                avg = sum(premiums) / len(premiums)
                maxp = premiums[-1]
                small_penalty = -80 if median < 120 else 0
                score = (
                    valid_cnt * 20 +
                    bigsum_cnt * 3 +
                    int(avg) + int(maxp) +
                    small_penalty +
                    header_bonus
                )

            if score > best_score:
                best_score = score
                best = {
                    "header_row": r_idx,
                    "c_program": c_program,
                    "c_sum": c_sum,
                    "c_prem": c_prem,
                }

    return best, best_score

def _classify_bta_template(parsed: Dict[str, Any]) -> str:
    """
    Kept for compatibility but no longer used to choose the path.
    """
    tables = parsed.get("tables") or []
    for tbl in tables:
        layout, score = _find_program_table_layout_scored(tbl)
        if layout and score > 0:
            return "offer"
    raw = (parsed.get("raw_text") or "")
    if re.search(r"BTA\s+APMAKSĀ", raw, re.IGNORECASE) or re.search(r"PROGRAMMA\s*[\-–]", raw, re.IGNORECASE):
        return "brochure"
    return "unknown"

def _extract_json_from_text(text: str) -> Optional[str]:
    try:
        json.loads(text); return text
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else None

# ----------------- Label-based recovery near an anchor -----------------
def _value_from_text_near(raw_text: str, anchor: str, label_regex: str,
                          min_val: Optional[float] = None,
                          max_val: Optional[float] = None,
                          window: int = 800) -> Optional[float]:
    """Search around `anchor` (program name) for a labeled number."""
    if not raw_text or not anchor:
        return None
    for m in re.finditer(re.escape(anchor), raw_text, flags=re.IGNORECASE):
        start = max(0, m.start() - window)
        end   = min(len(raw_text), m.end() + window)
        chunk = raw_text[start:end]
        lab = re.search(label_regex + r"[^\d%]{0,50}([0-9][0-9 \u00A0\.,]+)", chunk, re.IGNORECASE | re.DOTALL)
        if not lab:
            continue
        val = _num_safe(lab.group(1))
        if val is None:
            continue
        if (min_val is not None and val < min_val) or (max_val is not None and val > max_val):
            continue
        return val
    return None

# ----------------- Collect add-on values from text/tables -----------------
def _collect_bta_addons(parsed: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Collect add-on values used to fill bottom rows of main program(s).
    Now robustly reads Dentistry ('ZOBĀRSTNIECĪBA') sum from a separate sub-table:
    it detects the 'Apdrošinājuma summa' column via stacked headers and
    pulls the number from the dentistry row.
    """
    import unicodedata
    raw_text = parsed.get("raw_text") or ""
    tables = parsed.get("tables") or []
    out = {"dentistry_sum": None, "critical_sum": None, "vaccination_limit": None}

    def _canon(s: str) -> str:
        t = _norm_text(s)
        t = unicodedata.normalize("NFD", t)
        t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
        return t.lower()

    # ======================= ZOBĀRSTNIECĪBA (table-first, header-aware) =======================
    for tbl in tables:
        if not tbl:
            continue

        # locate rows that mention ZOBĀRSTNIECĪBA (accent-insensitive)
        dent_rows = []
        for r_idx, row in enumerate(tbl or []):
            row_txt = " ".join(_norm_text(c) for c in (row or []))
            if "zob" in _canon(row_txt) and "barstniec" in _canon(row_txt):  # matches ZOBĀRSTNIECĪBA variants
                dent_rows.append(r_idx)
        if not dent_rows:
            continue

        # find the "Apdrošinājuma summa" column using stacked headers (top ~6 rows)
        max_cols = max((len(r) for r in tbl if r), default=0)
        def stacked_header(col: int) -> str:
            parts = []
            for rr in range(0, min(6, len(tbl))):
                if col < len(tbl[rr]):
                    parts.append(_norm_text(tbl[rr][col]))
            return _canon(" ".join(parts))

        c_sum = None
        for c in range(max_cols):
            h = stacked_header(c)
            if ("apdrosinajuma" in h or "apdrošinājuma" in h) and "summa" in h:
                c_sum = c
                break

        # 1) Preferred: read the number from the dentistry row in the sum column
        found = False
        if c_sum is not None:
            for r_idx in dent_rows:
                if c_sum < len(tbl[r_idx]):
                    nums = _extract_numbers_no_percent(_norm_text(tbl[r_idx][c_sum]))
                    cand = [n for n in nums if 50 <= n <= 10000]
                    if cand:
                        out["dentistry_sum"] = max(cand)
                        found = True
                        break

        # 2) Fallback: if no header/number, take largest plausible in the dentistry row
        if not found:
            for r_idx in dent_rows:
                row_txt = " ".join(_norm_text(c) for c in (tbl[r_idx] or []))
                nums = _extract_numbers_no_percent(row_txt)
                cand = [n for n in nums if 50 <= n <= 10000]
                if cand:
                    out["dentistry_sum"] = max(cand)
                    found = True
                    break

        if found:
            break  # stop at first table that yields a value

    # ======================= KRITISKĀS SASLIMŠANAS (existing logic) =======================
    if out["critical_sum"] is None:
        def _flat(tbl_: List[List[str]]) -> str:
            return " ".join(" ".join(_norm_text(c) for c in (row or [])) for row in (tbl_ or []))

        crit_pat = re.compile(r"kri[tķ]isk\S*\s+saslim\S*", re.IGNORECASE)
        for tbl in tables:
            if not crit_pat.search(_flat(tbl)):
                continue
            max_cols = max((len(r) for r in tbl if r), default=0)
            header_cols: List[int] = []
            for r in range(min(8, len(tbl))):
                for c in range(max_cols):
                    parts = []
                    for rr in range(max(0, r - 2), min(len(tbl), r + 3)):
                        if c < len(tbl[rr]):
                            parts.append(_norm_text(tbl[rr][c]))
                    hdr = " ".join(parts).lower()
                    if ("apdrošinājuma" in hdr and "summa" in hdr) or ("apdrozininajuma" in hdr and "summa" in hdr):
                        header_cols.append(c)
            found = False
            for c in header_cols:
                col_nums: List[float] = []
                for rr in range(len(tbl)):
                    if c < len(tbl[rr]):
                        col_nums += _extract_numbers_no_percent(_norm_text(tbl[rr][c]))
                cand = [n for n in col_nums if 100 <= n <= 1_000_000]
                if cand:
                    out["critical_sum"] = max(cand)
                    found = True
                    break
            if found:
                break

    # ======================= RAW TEXT FALLBACKS (unchanged) =======================
    if out["dentistry_sum"] is None:
        m = re.search(
            r"ZOBĀRSTNIECĪBA[\s\S]{0,600}?Apdrošinājuma\s*summa[^\d%]{0,250}([0-9][0-9 \u00A0\.,]+)",
            raw_text, re.IGNORECASE)
        if m:
            n = _num_safe(m.group(1))
            if n and 50 <= n <= 10000:
                out["dentistry_sum"] = n
        else:
            m2 = re.search(r"ZOBĀRSTNIECĪBA[\s\S]{0,600}", raw_text, re.IGNORECASE)
            if m2:
                nums = _extract_numbers_no_percent(m2.group(0))
                nums = [x for x in nums if 50 <= x <= 10000]
                if nums:
                    out["dentistry_sum"] = max(nums)

    if out["critical_sum"] is None:
        m = re.search(
            r"KRITISK\S*\s+SASLIM\S*[\s\S]{0,1200}?Apdrošinājuma\s*summa[^\d%]{0,300}([0-9][0-9 \u00A0\.,]+)",
            raw_text, re.IGNORECASE)
        if m:
            n = _num_safe(m.group(1))
            if n and 100 <= n <= 1_000_000:
                out["critical_sum"] = n
        else:
            m2 = re.search(r"KRITISK\S*\s+SASLIM\S*[\s\S]{0,800}", raw_text, re.IGNORECASE)
            if m2:
                nums = _extract_numbers_no_percent(m2.group(0))
                nums = [x for x in nums if 100 <= x <= 1_000_000]
                if nums:
                    out["critical_sum"] = max(nums)

    m = re.search(r"\bVAKCINĀCIJA\b[\s\S]{0,400}?Limits?\s*([0-9][0-9 \u00A0\.,]+)", raw_text, re.IGNORECASE)
    if m:
        out["vaccination_limit"] = _num_safe(m.group(1))

    if os.getenv("DEBUG_BTA"):
        print("ADDONS:", out)

    return out

# -------- Payment terms: 'Polises apmaksas nosacījumi' (table-first, text-fallback)
def _extract_payment_terms(parsed: Dict[str, Any]) -> Optional[str]:
    """
    Return the value for 'Polises apmaksas nosacījumi' (e.g., 'Dalītie maksājumi bez piemaksas').
    """
    tables = parsed.get("tables") or []
    raw_text = parsed.get("raw_text") or ""

    label_re = re.compile(r"\bPolises\s+apmaksas\s+nosacījumi\b", re.IGNORECASE | re.DOTALL)

    def _clean_val(v: str) -> str:
        v = _norm_text(v)
        v = re.sub(r"^[\s\-–—:;|'\"•·.,]+|[\s\-–—:;|'\"•·.,]+$", "", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v

    _bad = re.compile(
        r"(starpnieks|pārdevēj|broker|brokerhouse|komisij|procent|apdrošināto\s+skaits|"
        r"grupu/variantu|kontaktperson|polises\s+darbības|pieteikum|atvērta\s+polise|"
        r"pelēk|uzlaboj|radinieku\s+apdrošināšana)", re.IGNORECASE
    )
    _good = re.compile(r"(dalīt|vienreizēj|bez\s+piemaks|pēc\s+cenr|pēc\s+rēķin|maksājum)", re.IGNORECASE)

    def _is_payment_phrase(v: str) -> bool:
        if not v or len(v) < 3: return False
        if "%" in v: return False
        if _bad.search(v): return False
        return bool(_good.search(v) or re.search(r"maks", v, re.IGNORECASE))

    def _split_lines(cell_text: str) -> list[str]:
        t = _norm_text(cell_text or "")
        t = re.sub(r"[•·▪●►]+", "\n", t)
        if "\n" not in t and "  " in t:
            t = re.sub(r"\s{2,}", "\n", t)
        return [p.strip() for p in re.split(r"[\r\n]+", t) if p.strip()]

    # ---- 1) TABLES (preferred) ----
    for tbl in tables:
        rows = [[_norm_text(c) for c in (row or [])] for row in (tbl or [])]
        nrows = len(rows)
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                if not label_re.search(cell or ""):
                    continue

                # (a) Same-cell "label: value"
                m_same = re.search(
                    r"Polises\s+apmaksas\s+nosacījumi\s*[:\-]\s*([^\r\n]+)",
                    cell or "", re.IGNORECASE,
                )
                if m_same:
                    val = _clean_val(m_same.group(1))
                    if _is_payment_phrase(val):
                        return val

                # (b) Simple neighbor to the RIGHT (same row)
                for cc in range(c + 1, min(c + 3, len(row))):
                    val = _clean_val(row[cc])
                    if _is_payment_phrase(val):
                        return val

                # (c) line-by-line alignment across merged columns
                left_lines = _split_lines(cell)
                right_cell = row[c + 1] if (c + 1) < len(row) else ""
                right_lines = _split_lines(right_cell)

                label_idxs = [i for i, t in enumerate(left_lines)
                              if re.search(r"polises\s+apmaksas\s+nosacīj", t, re.IGNORECASE)]
                for idx in (label_idxs or []):
                    if idx < len(right_lines):
                        candidate = _clean_val(right_lines[idx])
                        if _is_payment_phrase(candidate):
                            return candidate
                    if (r + 1) < nrows and (c + 1) < len(rows[r + 1]):
                        next_right_lines = _split_lines(rows[r + 1][c + 1])
                        if idx < len(next_right_lines):
                            candidate = _clean_val(next_right_lines[idx])
                            if _is_payment_phrase(candidate):
                                return candidate

                # (d) below in same column+1
                for rr in range(r + 1, min(r + 5, nrows)):
                    if (c + 1) < len(rows[rr]):
                        val = _clean_val(rows[rr][c + 1])
                        if _is_payment_phrase(val):
                            return val

                # (e) nearest valid phrase anywhere to the right within r..r+5
                best = None
                for rr in range(r, min(r + 6, nrows)):
                    for cc in range(c + 1, len(rows[rr])):
                        cand = _clean_val(rows[rr][cc])
                        if _is_payment_phrase(cand) and _good.search(cand):
                            best = cand
                            break
                    if best:
                        break
                if best:
                    return best

    # ---- 2) RAW TEXT fallback (tight) ----
    m = re.search(
        r"Polises\s+apmaksas\s+nosacījumi\s*[:\-]?\s*([^\r\n]{2,200})",
        raw_text, re.IGNORECASE,
    )
    if m:
        v = _clean_val(m.group(1))
        v = re.split(r"\b(īpašie\s+nosacījumi|piezīmes|starpnieks|pārdevēj|polises\s+darbības|grupu/variantu)\b",
                     v, 1, flags=re.IGNORECASE)[0].strip()
        if _is_payment_phrase(v):
            return v

    m2 = re.search(
        r"Polises\s+apmaksas\s+nosacījumi[\s\S]{0,200}"
        r"(dalīt[^\r\n]{0,120}|vienreizēj[^\r\n]{0,120}|bez\s+piemaks[^\r\n]{0,120}|pēc\s+cenr[^\r\n]{0,120}|pēc\s+rēķin[^\r\n]{0,120})",
        raw_text, re.IGNORECASE,
    )
    if m2:
        return _clean_val(m2.group(1))

    return None


# -------- Extract 'Pacientu iemaksa' once (table-first, then text)
def _extract_pacienta_iemaksa(parsed: Dict[str, Any]) -> Optional[str]:
    tables = parsed.get("tables") or []
    raw_text = parsed.get("raw_text") or ""

    label_re = re.compile(r"PACIENTA\s+IEMAKSA", re.IGNORECASE | re.DOTALL)
    pct_re = re.compile(r"(\d{1,3})\s*%")

    # 1) Tables
    for tbl in tables:
        for row in (tbl or []):
            cells = [_norm_text(c) for c in row]
            if not any(cells):
                continue
            joined = " ".join(cells)
            if label_re.search(joined):
                m_all = pct_re.findall(joined)
                if m_all:
                    return f"{m_all[-1]}%"
                ridx = tbl.index(row)
                for off in (1, 2):
                    if ridx + off < len(tbl):
                        j2 = " ".join(_norm_text(x) for x in tbl[ridx+off])
                        m2 = pct_re.search(j2)
                        if m2:
                            return f"{m2.group(1)}%"
                m3 = re.search(r"([0-9][0-9 \u00A0]*)\s*%", joined)
                if m3:
                    token = m3.group(1).replace(" ", "").replace("\u00A0","")
                    return f"{token}%"

    # 2) Raw text
    mtxt = re.search(r"PACIENTA\s+IEMAKSA[\s\S]{0,120}?(\d{1,3}\s*%)", raw_text, re.IGNORECASE)
    if mtxt:
        return _norm_text(mtxt.group(1)).replace(" ", "")

    return None

# -------- Helper: numeric value to the RIGHT of a label in tables --------
def _num_from_tables_right_of_label(parsed: Dict[str, Any], label_regex: str) -> Optional[float]:
    tables = parsed.get("tables") or []
    lab = re.compile(label_regex, re.IGNORECASE | re.DOTALL)

    for tbl in tables:
        for r, row in enumerate(tbl or []):
            cells = [_norm_text(c) for c in row]
            if not any(cells):
                continue
            for i, cell in enumerate(cells):
                if lab.search(cell):
                    for j in range(i + 1, len(cells)):
                        nums = _extract_numbers_no_percent(cells[j])
                        if nums:
                            return nums[0]
                    if r + 1 < len(tbl):
                        nxt = [_norm_text(c) for c in tbl[r + 1]]
                        nums = _extract_numbers_no_percent(" ".join(nxt[:3]))
                        if nums:
                            return nums[0]
    return None
    
# ======== COM generic helpers (paste right after _num_from_tables_right_of_label) ========

def _text_has_kw(raw_text: str, kw: str) -> bool:
    return bool(re.search(re.escape(kw), raw_text, re.IGNORECASE))

def _amount_near_kw(raw_text: str, kw: str, window: int = 600,
                    min_v: float = 0.0, max_v: float = 1_000_000) -> Optional[float]:
    """Find first plausible number near a keyword in raw text."""
    if not raw_text:
        return None
    for m in re.finditer(re.escape(kw), raw_text, flags=re.IGNORECASE):
        start = max(0, m.start() - window // 2)
        end = min(len(raw_text), m.end() + window)
        chunk = raw_text[start:end]
        nums = _extract_numbers_no_percent(chunk)
        cand = [n for n in nums if min_v <= n <= max_v]
        if cand:
            return cand[0]
    return None

def _find_value_in_tables_by_row(parsed: Dict[str, Any], row_regex: str) -> Optional[str]:
    """Return non-empty text found on the same row (prefer the rightmost non-empty cell)."""
    tables = parsed.get("tables") or []
    rgx = re.compile(row_regex, re.IGNORECASE | re.DOTALL)
    for tbl in tables:
        for row in (tbl or []):
            cells = [_norm_text(c) for c in (row or [])]
            if not any(cells):
                continue
            joined = " ".join(cells)
            if rgx.search(joined):
                vals = [c for c in cells if c and not rgx.search(c)]
                if vals:
                    return vals[-1]
    return None

def _amount_in_tables_by_row(parsed: Dict[str, Any], row_regex: str, max_pick: bool = True,
                             min_v: float = 0.0, max_v: float = 1_000_000) -> Optional[float]:
    """Find numeric value(s) on the same row; returns max or first."""
    tables = parsed.get("tables") or []
    rgx = re.compile(row_regex, re.IGNORECASE | re.DOTALL)
    for tbl in tables:
        for row in (tbl or []):
            cells = [_norm_text(c) for c in (row or [])]
            if not any(cells):
                continue
            joined = " ".join(cells)
            if rgx.search(joined):
                nums: List[float] = []
                for c in cells:
                    nums += _extract_numbers_no_percent(c)
                nums = [n for n in nums if min_v <= n <= max_v]
                if nums:
                    return max(nums) if max_pick else nums[0]
    return None

def _amount_from_tables_by_column_header(parsed: Dict[str, Any], header_regex: str,
                                         pick: str = "max", scan_rows: int = 30,
                                         min_v: float = 0.0, max_v: float = 1_000_000) -> Optional[float]:
    """Locate a column by header text (stacked allowed) and return a plausible number below it."""
    tables = parsed.get("tables") or []
    hdr = re.compile(header_regex, re.IGNORECASE | re.DOTALL)
    for tbl in tables:
        if not tbl:
            continue
        depth = min(8, len(tbl))
        max_cols = max((len(r) for r in tbl[:depth] if r), default=0)

        def stacked_header(col: int) -> str:
            parts = []
            for rr in range(depth):
                if col < len(tbl[rr]):
                    parts.append(_norm_text(tbl[rr][col]))
            return " ".join(parts)

        header_col = None
        for c in range(max_cols):
            if hdr.search(stacked_header(c)):
                header_col = c
                break
        if header_col is None:
            continue

        vals: List[float] = []
        for rr in range(depth, min(len(tbl), depth + scan_rows)):
            if header_col < len(tbl[rr]):
                nums = _extract_numbers_no_percent(_norm_text(tbl[rr][header_col]))
                for n in nums:
                    if min_v <= n <= max_v:
                        vals.append(n)
        if vals:
            if pick == "max":   return max(vals)
            if pick == "first": return vals[0]
    return None


def _amount_from_tables_by_header_in_titled_table(
    parsed: Dict[str, Any],
    table_title_regex: str,
    header_regex: str,
    pick: str = "first",
    scan_rows: int = 40,
    min_v: float = 0.0,
    max_v: float = 1_000_000,
) -> Optional[float]:
    """Find value in a specific (titled) table under a stacked column header."""
    tables = parsed.get("tables") or []
    title_rgx = re.compile(table_title_regex, re.IGNORECASE | re.DOTALL)
    hdr_rgx = re.compile(header_regex, re.IGNORECASE | re.DOTALL)

    for tbl in tables:
        if not tbl:
            continue

        # Build a 'title' from the first 2 rows (common in COM PDFs)
        first = " ".join(_norm_text(c) for c in (tbl[0] or []))
        second = " ".join(_norm_text(c) for c in (tbl[1] or [])) if len(tbl) > 1 else ""
        title = (first + " " + second).strip()
        if not title_rgx.search(title):
            continue

        # Locate the header column by stacked header (top 5 rows)
        depth = min(5, len(tbl))
        max_cols = max((len(r) for r in tbl[:depth] if r), default=0)

        def stacked_header(col: int) -> str:
            return " ".join(_norm_text(tbl[r][col]) for r in range(depth) if col < len(tbl[r]))

        header_col = None
        for c in range(max_cols):
            if hdr_rgx.search(stacked_header(c)):
                header_col = c
                break
        if header_col is None:
            continue

        # Collect plausible values below that header
        vals: List[float] = []
        for rr in range(depth, min(len(tbl), depth + scan_rows)):
            if header_col < len(tbl[rr]):
                nums = _extract_numbers_no_percent(_norm_text(tbl[rr][header_col]))
                for n in nums:
                    if min_v <= n <= max_v:
                        vals.append(n)
        if vals:
            return vals[0] if pick == "first" else (max(vals) if pick == "max" else vals[-1])
    return None


def _section_has_phrase(raw_text: str, section_regex: str, phrase_regex: str, window: int = 1500) -> bool:
    """Look for phrase within a limited window after a section header."""
    if not raw_text:
        return False
    srx = re.compile(section_regex, re.IGNORECASE | re.DOTALL)
    prx = re.compile(phrase_regex, re.IGNORECASE | re.DOTALL)
    for m in srx.finditer(raw_text):
        start = m.end()
        end = min(len(raw_text), start + window)
        if prx.search(raw_text[start:end]):
            return True
    return False



# -------- Fill remaining BTA detail rows (table-first, then text) --------
def fill_bta_detail_fields(features: dict, parsed: Dict[str, Any]) -> dict:
    raw_text = parsed.get("raw_text") or ""
    tables = parsed.get("tables") or []

    # ---------- BTA2 HARD OVERRIDE (NO filename heuristics) ----------
    def _is_bta2() -> bool:
        def _norm(x):
            return (str(x or "")).strip().lower()
        hints = {
            _norm(features.get("company_hint")),
            _norm(parsed.get("company_hint")),
            _norm(features.get("brochure_code")),
            _norm(parsed.get("brochure_code")),
        }
        return any(h in {"bta2", "bta 2", "bta_2"} for h in hints)

    IS_BTA2 = _is_bta2()

    # Freeze premium fields for BTA2
    if IS_BTA2:
        if "Pamatpolises prēmija 1 darbiniekam, EUR" in features:
            features["Pamatpolises prēmija 1 darbiniekam, EUR"] = "-"
        if "premium_eur" in features:
            features["premium_eur"] = "-"

    def set_num_from_label(key: str, label_regex: str,
                           min_val: float | None = None, max_val: float | None = None):
        if key not in features:
            return
        m = re.search(label_regex + r"[^\d%]{0,80}([0-9][0-9 \u00A0\.,]+)",
                      raw_text, re.IGNORECASE | re.DOTALL)
        if not m:
            return
        v = _num_safe(m.group(1))
        if v is None:
            return
        if (min_val is not None and v < min_val) or (max_val is not None and v > max_val):
            return
        features[key] = v

    # ---------- VAKCINĀCIJA helpers ----------
    def _vaccine_limit_from_tables() -> Optional[float]:
        lab = re.compile(r"\bVAKCIN", re.IGNORECASE)
        for tbl in tables:
            for r, row in enumerate(tbl or []):
                cells = [_norm_text(c) for c in (row or [])]
                if not any(cells):
                    continue
                label_cols = [c for c, txt in enumerate(cells) if lab.search(txt or "")]
                if not label_cols:
                    continue
                c_label = label_cols[0]
                for cc in range(c_label + 1, len(cells)):
                    txt = cells[cc]
                    if not txt:
                        continue
                    m = re.search(r"limit\w*[^\d%]{0,40}([0-9][0-9 \u00A0\.,]+)\s*(?:EUR)?", txt, re.IGNORECASE)
                    if m:
                        v = _num_safe(m.group(1))
                        if v is not None and 0 < v <= 10000:
                            return v
                amount_col = None
                for cc in range(c_label + 1, len(row)):
                    has_nums_next = False
                    for rr in range(r + 1, min(r + 12, len(tbl))):
                        if cc < len(tbl[rr]):
                            nums = _extract_numbers_no_percent(_norm_text(tbl[rr][cc]))
                            if nums:
                                has_nums_next = True
                                break
                    if has_nums_next:
                        amount_col = cc
                        break
                if amount_col is None:
                    continue
                total = 0.0
                seen_any = False
                for rr in range(r + 1, len(tbl)):
                    row2 = tbl[rr] or []
                    left_blob = " ".join(_norm_text(x) for x in row2[:max(1, c_label + 1)])
                    if (not any(_norm_text(x) for x in row2)) or re.search(
                        r"^[A-ZĀČĒĢĪĶĻŅŌŖŠŪŽ][A-ZĀČĒĢĪĶĻŅŌŖŠŪŽ0-9 \-_/]{3,}$", left_blob
                    ):
                        if seen_any:
                            break
                        else:
                            continue
                    if amount_col < len(row2):
                        nums = _extract_numbers_no_percent(_norm_text(row2[amount_col]))
                        cand = [n for n in nums if 0 < n <= 10000]
                        if cand:
                            total += cand[0]
                            seen_any = True
                    else:
                        if seen_any:
                            break
                if seen_any:
                    return total
        return None

    def _vaccine_limit_from_text() -> Optional[float]:
        for m in re.finditer(r"\bVAKCINĀCIJA\b", raw_text, re.IGNORECASE):
            start = max(0, m.start() - 200)
            end   = min(len(raw_text), m.end() + 600)
            window = raw_text[start:end]
            mlim = re.search(r"limit\w*[^\d%]{0,40}([0-9][0-9 \u00A0\.,]+)\s*(?:EUR)?", window, re.IGNORECASE)
            if mlim:
                v = _num_safe(mlim.group(1))
                if v is not None and 0 < v <= 10000:
                    return v
        for m in re.finditer(r"\bVAKCINĀCIJA\b", raw_text, re.IGNORECASE):
            start = m.end()
            stop = min(len(raw_text), start + 1200)
            block = raw_text[start:stop]
            vals = []
            for line in re.split(r"[\r\n]+", block):
                nums = _extract_numbers_no_percent(line)
                cand = [n for n in nums if 0 < n <= 10000]
                if cand:
                    vals.append(cand[0])
            if vals:
                return float(sum(vals))
        return None

    # ---------- Mājas vizītes ----------
    if "Maksas ģimenes ārsta mājas vizītes, limits EUR" in features:
        key_home = "Maksas ģimenes ārsta mājas vizītes, limits EUR"
        v = None
        # Prefer the long phrase that includes transport services (covers 35.00 for BTA2 V2+ and 40.00 for BTA V3)
        v = _num_from_tables_right_of_label(
            parsed,
            r"Mājas\s+vizītes[^\r\n]{0,200}?transporta\s+pakalpojumi"
        )
        if v is None:
            m = re.search(
                r"Mājas\s+vizītes[^\r\n]{0,200}?transporta\s+pakalpojumi[^\d%]{0,60}"
                r"([0-9][0-9 \u00A0\.,]+)",
                raw_text, re.IGNORECASE
            )
            if m:
                v = _num_safe(m.group(1))
        # Fallback to generic
        if v is None:
            v = _num_from_tables_right_of_label(parsed, r"\bMājas\s+vizītes\b")
        if v is None:
            set_num_from_label(key_home, r"\bMājas\s+vizītes\b", 1, 5000)
        if v is not None:
            features[key_home] = v

    # ---------- Maksa ģimenes ārsts / internists / pediatrs ----------
    fam_pat = r"Maksa\s+ģimenes\s+ārsts.*?(?:Internists|Pediatrs)"
    v_fam = _num_from_tables_right_of_label(parsed, fam_pat)
    if v_fam is not None:
        if "Maksas ģimenes ārsta, internista, terapeita un pediatra konsultācija, limits EUR" in features:
            features["Maksas ģimenes ārsta, internista, terapeita un pediatra konsultācija, limits EUR"] = v_fam
        if "Maksas ārsta-specialista konsultācija, limits EUR" in features:
            features["Maksas ārsta-specialista konsultācija, limits EUR"] = v_fam
    else:
        set_num_from_label("Maksas ģimenes ārsta, internista, terapeita un pediatra konsultācija, limits EUR", fam_pat)
        if features.get("Maksas ģimenes ārsta, internista, terapeita un pediatra konsultācija, limits EUR") not in ("-", None):
            features["Maksas ārsta-specialista konsultācija, limits EUR"] = \
                features["Maksas ģimenes ārsta, internista, terapeita un pediatra konsultācija, limits EUR"]

    # ---------- Profesora, docenta… ----------
    prof_key = "Profesora, docenta, internista konsultācija, limits EUR"

    # Prefer FIRST-TIME consultation phrase if present (covers 50.00 for BTA2 V2+ and 45.00 for BTA V3)
    v_prof_first = _num_from_tables_right_of_label(
        parsed,
        r"Pirmreizēj\w*\s+konsultāc\w*[^\r\n]{0,200}?profesor[^\r\n]{0,200}?docent[^\r\n]{0,200}?augst\w+\s+kvalifikāc\w+\s+speciālist"
    )
    if v_prof_first is None:
        m_first = re.search(
            r"Pirmreizēj\w*\s+konsultāc\w*[^\r\n]{0,200}?profesor[^\r\n]{0,200}?docent[^\r\n]{0,200}?augst\w+\s+kvalifikāc\w+\s+speciālist[^\d%]{0,60}"
            r"([0-9][0-9 \u00A0\.,]+)",
            raw_text, re.IGNORECASE
        )
        if m_first:
            v_prof_first = _num_safe(m_first.group(1))

    if v_prof_first is not None:
        features[prof_key] = v_prof_first
    else:
        # If first-time is absent, try repeated consultation phrase (e.g., 45.00 in some docs)
        v_prof_repeat = _num_from_tables_right_of_label(
            parsed,
            r"Atkārtot\w*\s+konsultāc\w*[^\r\n]{0,200}?profesor[^\r\n]{0,200}?docent[^\r\n]{0,200}?augst\w+\s+kvalifikāc\w+\s+speciālist"
        )
        if v_prof_repeat is None:
            m_rep = re.search(
                r"Atkārtot\w*\s+konsultāc\w*[^\r\n]{0,200}?profesor[^\r\n]{0,200}?docent[^\r\n]{0,200}?augst\w+\s+kvalifikāc\w+\s+speciālist[^\d%]{0,60}"
                r"([0-9][0-9 \u00A0\.,]+)",
                raw_text, re.IGNORECASE
            )
            if m_rep:
                v_prof_repeat = _num_safe(m_rep.group(1))
        if v_prof_repeat is not None:
            features[prof_key] = v_prof_repeat
        else:
            # Generic fallback (keeps your existing logic)
            v_prof_generic = _num_from_tables_right_of_label(parsed, r"Profesora[\s,]*docenta[\s,]*.*?konsultāc")
            if v_prof_generic is not None:
                features[prof_key] = v_prof_generic
            else:
                set_num_from_label(
                    prof_key,
                    r"(?:Profesora[\s,]*docenta[\s,]*.*?konsultāc|Profesora[\s,]*docenta.*?augstākās\s+kvalifikācijas\s+speciālistu\s+konsultāc)",
                    1, 20000
                )

    # ---------- Homeopāts & Psihoterapeits ----------
    def _two_x_present(stem: str) -> bool:
        pats = [
            rf"2\s*\(\s*divas?\s*\)\s*{stem}",
            rf"{stem}[^\n\r]{{0,120}}2\s*\(\s*divas?\s*\)\s*konsult",
            rf"\b2x\b[^\n\r]{{0,60}}{stem}",
            rf"{stem}[^\n\r]{{0,60}}\b2x\b",
            rf"\b2\b[^\n\r]{{0,20}}{stem}\b.*?\bkonsult",
        ]
        return any(re.search(p, raw_text, re.IGNORECASE) for p in pats)

    if "Homeopāts" in features and (features["Homeopāts"] in ("-", None, "")):
        if _two_x_present(r"homeop[aā]t[aā]?"):
            features["Homeopāts"] = "2x"
    if "Psihoterapeits" in features and (features["Psihoterapeits"] in ("-", None, "")):
        if _two_x_present(r"psihoterap(?:eits|eita)"):
            features["Psihoterapeits"] = "2x"

    # ---------- Sporta ārsts ----------
    if "Sporta ārsts" in features and (features["Sporta ārsts"] in ("-", None, "")):
        features["Sporta ārsts"] = "v" if re.search(r"\bSporta\s+ārsts\b", raw_text, re.IGNORECASE) else "-"

    # ---------- Fizikālā terapija ----------
    if "Fizikālā terapija" in features and (features["Fizikālā terapija"] in ("-", None, "")):
        m = re.search(r"FIZIKĀLĀS\s+TERAPIJAS\s+PROCEDŪRAS[^\r\n]*?([0-9]+)\s*x", raw_text, re.IGNORECASE)
        if not m:
            m = re.search(r"fizik[āa]l[āa]s\s+terapijas\s+procedūr\w*[^\r\n]{0,80}?līdz\s*([0-9]+)\s*reiz", raw_text, re.IGNORECASE)
        if m:
            features["Fizikālā terapija"] = f"{m.group(1)}x periodā"

    # ---------- Vakcinācija, limits EUR ----------
    if "Vakcinācija, limits EUR" in features:
        v_limit = _vaccine_limit_from_tables()
        if v_limit is None:
            v_limit = _vaccine_limit_from_text()
        if v_limit is not None:
            features["Vakcinācija, limits EUR"] = v_limit

    # ---------- Neatliekamā palīdzība ----------
    key_np = "Neatliekamā palīdzība valsts un privātā (limits privātai, EUR)"
    if key_np in features and (features[key_np] in ("-", None, "")):
        v_emerg = _num_from_tables_right_of_label(
            parsed,
            r"Valsts\s+un\s+privāt[āa]\s+neatliekam[āa]\s+medicīnisk[āa]\s+palīdzība"
        )
        if v_emerg is None:
            v_emerg = _num_from_tables_right_of_label(
                parsed,
                r"\bNEATLIEKAM[ĀA]\s+MEDICĪNISK[ĀA]\s+PALĪDZĪBA\b"
            )
        if v_emerg is not None:
            features[key_np] = v_emerg
        else:
            set_num_from_label(key_np, r"t\.\s*sk\.\s*privāt[āa]", 1, 100000)

    # ---------- Maksas stacionārā rehabilitācija ----------
    key_rehab = "Maksas stacionārā rehabilitācija, limits EUR"
    if key_rehab in features and (features[key_rehab] in ("-", None, "")):
        v_rehab = _num_from_tables_right_of_label(
            parsed,
            r"STACIONĀR\S*\s+REHABILITĀCIJ\S*\s+MAKSAS\s+PAKALPOJUM\S*"
        )
        if v_rehab is None:
            v_rehab = _num_from_tables_right_of_label(parsed, r"Maksas\s+stacionār(?:ā|as)\s+rehabilitāc")
        if v_rehab is None:
            m_txt = re.search(
                r"STACIONĀR\S*\s+REHABILITĀCIJ\S*[\s\S]{0,200}?Limits?\s*([0-9][0-9 \u00A0\.,]+)",
                raw_text, re.IGNORECASE
            )
            if not m_txt:
                m_txt = re.search(
                    r"Maksas\s+stacionār(?:ā|as)\s+rehabilitāc\w*[\s\S]{0,200}?Limits?\s*([0-9][0-9 \u00A0\.,]+)",
                    raw_text, re.IGNORECASE
                )
            if m_txt:
                v_tmp = _num_safe(m_txt.group(1))
                if v_tmp is not None and 1 <= v_tmp <= 1_000_000:
                    v_rehab = v_tmp
        if v_rehab is not None:
            features[key_rehab] = v_rehab

    # ---------- Onkoloģiskā/hematoloģiskā ārstēšana ----------
    if "Maksas onkoloģiskā, hematoloģiskā ārstēšana" in features:
        v_onk = _num_from_tables_right_of_label(parsed, r"Maksas\s+onkoloģiskā.*?hematoloģiskā\s+ārstēšana")
        if v_onk is None:
            v_onk = _num_from_tables_right_of_label(parsed, r"onkoloģ.*?hematoloģ.*?ārstēšan")
        if v_onk is None:
            set_num_from_label(
                "Maksas onkoloģiskā, hematoloģiskā ārstēšana",
                r"(?:Maksas\s+onkoloģiskā.*?hematoloģiskā\s+ārstēšan|onkoloģ(?:iskā)?\s+.*?hematoloģ(?:iskā)?\s+ārstēšan)",
                1, 1_000_000
            )
        else:
            features["Maksas onkoloģiskā, hematoloģiskā ārstēšana"] = v_onk

    # ---------- MAKSAS STACIONĀRS ----------
    if "Maksas stacionārie pakalpojumi, limits EUR" in features:
        v_ms = _num_from_tables_right_of_label(parsed, r"\bMAKSAS\s+STACIONĀRS\b")
        if v_ms is None:
            m_txt = re.search(
                r"STACIONĀR\S*\s+MAKSAS\s+PAKALPOJUM\S*[\s\S]{0,220}?Limits?\s*([0-9][0-9 \u00A0\.,]+)",
                raw_text, re.IGNORECASE
            )
            if m_txt:
                v_t = _num_safe(m_txt.group(1))
                if v_t is not None and 1 <= v_t <= 1_000_000:
                    v_ms = v_t
        if v_ms is None:
            set_num_from_label("Maksas stacionārie pakalpojumi, limits EUR",
                               r"\bMAKSAS\s+STACIONĀRS\b", 1, 1_000_000)
        else:
            features["Maksas stacionārie pakalpojumi, limits EUR"] = v_ms

    # ---------- Enforce BTA2 premium mask again ----------
    if IS_BTA2:
        if "Pamatpolises prēmija 1 darbiniekam, EUR" in features:
            features["Pamatpolises prēmija 1 darbiniekam, EUR"] = "-"
        if "premium_eur" in features:
            features["premium_eur"] = "-"

    return features



def _apply_bta_bottom_rules(features: dict, addons: Dict[str, Optional[float]]) -> dict:
    """
    EXACTLY your requested bottom rows for BTA main programs.
    """
    if "Piemaksa par plastikāta kartēm, EUR" in features:
        features["Piemaksa par plastikāta kartēm, EUR"] = 5
    if "Zobārstniecība ar 50% atlaidi (pamatpolise)" in features:
        features["Zobārstniecība ar 50% atlaidi (pamatpolise)"] = "-"
    if "ONLINE ārstu konsultācijas" in features:
        features["ONLINE ārstu konsultācijas"] = "100%"
    if "Obligātās veselības pārbaudes, limits EUR" in features:
        features["Obligātās veselības pārbaudes, limits EUR"] = "100%"
    if "Laboratoriskie izmeklējumi" in features:
        features["Laboratoriskie izmeklējumi"] = "v"
    if "Maksas diagnostika, piem., rentgens, elektrokradiogramma, USG, utml." in features:
        features["Maksas diagnostika, piem., rentgens, elektrokradiogramma, USG, utml."] = "v"
    if "Augsto tehnoloģiju izmeklējumi, piem., MRG, CT, limits (reižu skaits vai EUR)" in features:
        features["Augsto tehnoloģiju izmeklējumi, piem., MRG, CT, limits (reižu skaits vai EUR)"] = "v"
    if "Ārstnieciskās manipulācijas" in features:
        features["Ārstnieciskās manipulācijas"] = "v"
    if "Medicīniskās izziņas" in features:
        features["Medicīniskās izziņas"] = "v"
    if "Procedūras" in features:
        features["Procedūras"] = "v"
    if "Maksas grūtnieču aprūpe" in features:
        features["Maksas grūtnieču aprūpe"] = "v"
    if "Sports" in features:
        features["Sports"] = "-"

    if "Ambulatorā rehabilitācija" in features:
        features["Ambulatorā rehabilitācija"] = "-"

    if "Ambulatorā rehabilitācija (pp)" in features:
        features["Ambulatorā rehabilitācija (pp)"] = "100 EUR limits"

    if "Maksas stacionārie pakalpojumi, limits EUR (pp)" in features:
        features["Maksas stacionārie pakalpojumi, limits EUR (pp)" ] = "ir iekļauts"

    if "Medikamenti ar 50% atlaidi" in features:
        features["Medikamenti ar 50% atlaidi"] = "70 EUR limits"

    if "Zobārstniecība ar 50% atlaidi, apdrošinājuma summa (pp)" in features:
        features["Zobārstniecība ar 50% atlaidi, apdrošinājuma summa (pp)"] = addons.get("dentistry_sum") or "-"
    if "Kritiskās saslimšanas" in features:
        features["Kritiskās saslimšanas"] = addons.get("critical_sum") or "-"

    if "Vakcinācija pret ērčiem un gripu" in features:
        vac = addons.get("vaccination_limit")
        if vac in (None, "-"):
            vac = features.get("Vakcinācija, limits EUR", "-")
        features["Vakcinācija pret ērčiem un gripu"] = vac

    return features
    
# ========================= COMPENSA (COM_VA) =========================

def com_programs(parsed: Dict[str, Any]) -> List[ProgramModel]:
    """
    Extract Compensa values per mapping:
      - program_code & Programmas nosaukums: row 'Pamatprogrammā iekļautie maksas pakalpojumi'
      - base_sum_eur: column 'Apdrošinājuma summa vienai personai, EUR'
      - premium_eur: column 'Prēmija vienai personai, EUR'
      - many features via row/keyword lookups + COM hardcoded flags.
    """
    raw_text = parsed.get("raw_text") or ""

    # ---- program_code / Programmas nosaukums ----
    prog_label_re = r"Pamatprogramm[āa]\s+iekļautie\s+maksas\s+pakalpojumi"
    program_code_text = _find_value_in_tables_by_row(parsed, prog_label_re)
    if not program_code_text:
        m = re.search(prog_label_re + r"[^\r\n]{0,80}?([A-Z0-9+/ ]{3,})", raw_text, re.IGNORECASE)
        program_code_text = _norm_text(m.group(1)) if m else "Pamatprogramma"
    program_code_text = _norm_text(program_code_text)

# ---- base_sum_eur ----
# First try the exact green sentence variant (appears only once in text)
    m_green = re.search(
        r"Pamatprogrammas\s+apdrošinājuma\s+summa\s+vienai\s+personai[^\d%]{0,40}([0-9][0-9 \u00A0\.,]+)",
        raw_text, re.IGNORECASE
    )
    base_sum = _num_safe(m_green.group(1)) if m_green else None
    
    if not base_sum:
        base_sum = (
            _amount_from_tables_by_header_in_titled_table(
                parsed,
                r"PAMATPROGRAMMA\s+UN\s+PAPILDU\s+SEGUMS",
                r"Apdrošinājuma\s+summa\s+vienai\s+personai,\s*EUR",
                pick="first", min_v=1, max_v=10_000_000
            )
            or _amount_from_tables_by_column_header(
                parsed, r"Apdrošinājuma\s+summa\s+vienai\s+personai,\s*EUR",
                pick="first", min_v=1, max_v=10_000_000
            )
            # Require min_v=100 so we don’t accidentally grab “1.” section numbers
            or _amount_near_kw(
                raw_text,
                "Pamatprogrammas apdrošinājuma summa vienai personai",
                min_v=100, max_v=10_000_000
            )
            or _amount_near_kw(
                raw_text,
                "Apdrošinājuma summa vienai personai, EUR",
                min_v=100, max_v=10_000_000
            )
            or 0.0
        )


# ---- premium_eur ----
    premium = None
    try:
        tables = parsed.get("tables") or []
        title_ok  = re.compile(r"PAMATPROGRAMMA\s+UN\s+PAPILDU\s+SEGUMS", re.IGNORECASE)
        title_bad = re.compile(r"\bPAPILDPROGRAMMAS\b", re.IGNORECASE)
        hdr_prem  = re.compile(r"Prēmija\s+vienai\s*[\r\n ]*personai,\s*EUR", re.IGNORECASE)
        hdr_sum   = re.compile(r"Apdrošinājuma\s+summa\s+vienai\s*[\r\n ]*personai,\s*EUR", re.IGNORECASE)
    
        def _stacked_header(tbl, depth, c):
            parts = []
            for rr in range(depth):
                if c < len(tbl[rr]):
                    parts.append(_norm_text(tbl[rr][c]))
            return " ".join(parts)
    
        def _first_number_under(tbl, col, start_row, scan_rows=40, min_v=50, max_v=10000):
            for rr in range(start_row, min(len(tbl), start_row + scan_rows)):
                if col < len(tbl[rr]):
                    nums = _extract_numbers_no_percent(_norm_text(tbl[rr][col]))
                    for n in nums:
                        if min_v <= n <= max_v:
                            return n
            return None
    
        # -------- (1) Titled table + neighbor peek (for skinny premium table)
        for i, tbl in enumerate(tables):
            if not tbl:
                continue
    
            # build a loose title out of first few rows
            title_blob = " ".join(
                " ".join(_norm_text(c) for c in (tbl[r] or []))
                for r in range(0, min(4, len(tbl)))
            )
            if not title_ok.search(title_blob) or title_bad.search(title_blob):
                continue
    
            depth = min(8, len(tbl))
            max_cols = max((len(r) for r in tbl[:depth] if r), default=0)
    
            prem_col = None
            for c in range(max_cols):
                if hdr_prem.search(_stacked_header(tbl, depth, c)):
                    prem_col = c
                    break
    
            # Try inside the same table first
            if prem_col is not None:
                pval = _first_number_under(tbl, prem_col, depth)
                if pval is not None:
                    premium = pval
                    break
    
            # <— NEW: neighbor tables (skinny premium column split out by the PDF)
            # Right neighbor
            if i + 1 < len(tables) and tables[i + 1]:
                tR = tables[i + 1]
                depthR = min(8, len(tR))
                max_colsR = max((len(r) for r in tR[:depthR] if r), default=0)
                for c in range(max_colsR):
                    if hdr_prem.search(_stacked_header(tR, depthR, c)):
                        pval = _first_number_under(tR, c, depthR)
                        if pval is not None:
                            premium = pval
                            break
                if premium is not None:
                    break
    
            # Left neighbor (rare, but cheap to try)
            if i - 1 >= 0 and tables[i - 1]:
                tL = tables[i - 1]
                depthL = min(8, len(tL))
                max_colsL = max((len(r) for r in tL[:depthL] if r), default=0)
                for c in range(max_colsL):
                    if hdr_prem.search(_stacked_header(tL, depthL, c)):
                        pval = _first_number_under(tL, c, depthL)
                        if pval is not None:
                            premium = pval
                            break
                if premium is not None:
                    break
    
        # -------- (2) Header-pair fallback (unchanged) ...
        if premium is None:
            for tbl in tables:
                if not tbl:
                    continue
                top_blob = " ".join(
                    " ".join(_norm_text(c) for c in (tbl[i] or []))
                    for i in range(0, min(5, len(tbl)))
                )
                if title_bad.search(top_blob):
                    continue
                depth = min(8, len(tbl))
                max_cols = max((len(r) for r in tbl[:depth] if r), default=0)
                prem_col = sum_col = None
                for c in range(max_cols):
                    head = _stacked_header(tbl, depth, c)
                    if prem_col is None and hdr_prem.search(head): prem_col = c
                    if sum_col  is None and hdr_sum.search(head):  sum_col  = c
                if prem_col is None or sum_col is None:
                    continue
                pval = _first_number_under(tbl, prem_col, depth)
                if pval is not None:
                    premium = pval
                    break
    
    except Exception:
        pass


    # -------- (3) Last-resort fallbacks
    if premium is None:
        premium = (
            _amount_from_tables_by_header_in_titled_table(
                parsed,
                r"PAMATPROGRAMMA\s+UN\s+PAPILDU\s+SEGUMS",
                r"Prēmija\s+vienai\s*[\r\n ]*personai,\s*EUR",
                pick="first", min_v=50, max_v=100_000
            )
            or _amount_from_tables_by_column_header(
                parsed, r"Prēmija\s+vienai\s*[\r\n ]*personai,\s*EUR",
                pick="first", min_v=50, max_v=100_000
            )
            or _amount_near_kw(raw_text, "Prēmija vienai personai, EUR", min_v=50, max_v=100_000)
            or 0.0
        )


    # ---- features ----
    feats = {
        "Programmas nosaukums": program_code_text,
        "Programmas kods": program_code_text,
        "Apdrošinājuma summa pamatpolisei, EUR": base_sum or "-",
        "Pamatpolises prēmija 1 darbiniekam, EUR": premium or "-",

        # COM_VA hardcoded per your spec
        "Pacientu iemaksa": "100%",
        "ONLINE ārstu konsultācijas": "v",
        "Laboratoriskie izmeklējumi": "v",
        "Maksas diagnostika, piem., rentgens, elektrokradiogramma, USG, utml.": "v",
        "Obligātās veselības pārbaudes, limits EUR": "100%",
        "Ārstnieciskās manipulācijas": "v",
        "Medicīniskās izziņas": "v",
        "Procedūras": "v",
    }

    # Row / keyword lookups
    feats["Maksas ģimenes ārsta mājas vizītes, limits EUR"] = (
        _amount_in_tables_by_row(parsed, r"Ģimenes\s+ārstu,\s*internistu\s+un\s+pediatru\s+mājas\s+vizīte",
                                 min_v=1, max_v=10_000) or
        _amount_near_kw(raw_text, "mājas vizīte", max_v=10_000) or "-"
    )

    fam_cons_val = (
        _amount_in_tables_by_row(parsed, r"Ģimenes\s+ārstu.*?(internistu).*?(pediatru).*?konsultāc", min_v=1, max_v=10_000) or
        feats["Maksas ģimenes ārsta mājas vizītes, limits EUR"]
    )
    feats["Maksas ģimenes ārsta, internista, terapeita un pediatra konsultācija, limits EUR"] = fam_cons_val or "-"

    feats["Maksas ārsta-specialista konsultācija, limits EUR"] = (
        _amount_in_tables_by_row(parsed,
            r"ārstu\s+speciālistu.*?konsultācijas|dermatologa|ginekologa|ķirurga|neirologa|traumatologa",
            min_v=1, max_v=10_000) or "-"
    )

    feats["Profesora, docenta, internista konsultācija, limits EUR"] = (
        _amount_in_tables_by_row(parsed, r"Medicīnas\s+profesoru\s+un\s+docentu\s+konsultāc", min_v=1, max_v=10_000) or
        _amount_near_kw(raw_text, "profesoru un docentu konsultāc", max_v=10_000) or "-"
    )

    def _mk_times_or_v(kw: str) -> str:
        t = "-"
        if _text_has_kw(raw_text, kw) or _find_value_in_tables_by_row(parsed, kw):
            m = re.search(r"(vienu|1|2|3|4|5)\s+reiz", raw_text, re.IGNORECASE)
            if m:
                token = m.group(1).lower()
                t = "1x" if token in ("vienu", "1") else f"{token}x"
            else:
                t = "v"
        return t

    feats["Homeopāts"] = _mk_times_or_v(r"homeop[aā]t")
        # Mark as 'v' only if the PAMATPROGRAMMA section explicitly mentions the consultation phrase
    has_psy = _section_has_phrase(
        raw_text,
        r"\bPAMATPROGRAMMA\b",
        r"Psihologa,\s*psihoterapeita\s+vai\s+psihiatra\s+konsultāc"
    ) or bool(_find_value_in_tables_by_row(parsed, r"Psihologa,\s*psihoterapeita\s+vai\s+psihiatra\s+konsultāc"))
    feats["Psihoterapeits"] = "v" if has_psy else "-"
    feats["Sporta ārsts"] = "v" if _text_has_kw(raw_text, "sporta ārsta konsultācija") or \
                              _find_value_in_tables_by_row(parsed, r"sporta\s+ārsta") else "-"

    feats["Augsto tehnoloģiju izmeklējumi, piem., MRG, CT, limits (reižu skaits vai EUR)"] = (
        _amount_in_tables_by_row(parsed, r"Skaitļotājtomogrāfijas\s*\(CT\)\s*izmeklējumi", min_v=1, max_v=100_000) or "v"
    )

    # Fizikālā terapija – try to assemble "X reizes, Y%"
    fiz_text = raw_text if _text_has_kw(raw_text, "Fizikālā terapija") else ""
    times = re.search(r"(\d+)\s*reiz", fiz_text, re.IGNORECASE)
    pct = re.search(r"(\d{1,3})\s*%", fiz_text)
    if times and pct:
        feats["Fizikālā terapija"] = f"{times.group(1)} reizes, {pct.group(1)}%"
    else:
        feats["Fizikālā terapija"] = _find_value_in_tables_by_row(parsed, r"Fizik[āa]l[āa]\s+terapij") or "-"

    feats["Vakcinācija, limits EUR"] = (
        _amount_in_tables_by_row(parsed, r"Vakc(in|ināc)", min_v=1, max_v=10_000) or
        _amount_near_kw(raw_text, "Vakcinācija", max_v=10_000) or "-"
    )

    feats["Maksas grūtnieču aprūpe"] = "v" if (
        _text_has_kw(raw_text, "Grūtnieču aprūpe") or _find_value_in_tables_by_row(parsed, r"Grūtnieču\s+aprūpe")
    ) else "-"

    feats["Maksas onkoloģiskā, hematoloģiskā ārstēšana"] = "v" if (
        _text_has_kw(raw_text, "Onkologa ārstēšana") or _find_value_in_tables_by_row(parsed, r"Onkolog")
    ) else "-"

    feats["Neatliekamā palīdzība valsts un privātā (limits privātai, EUR)"] = (
        _amount_in_tables_by_row(parsed, r"Neatliekam[āa]\s+medicīnisk[āa]\s+pal[īi]dz[īi]ba.*privāt", min_v=1, max_v=1_000_000) or
        _amount_near_kw(raw_text, "privātā neatliekamā medicīniskā palīdzība", max_v=1_000_000) or "-"
    )

    feats["Maksas stacionārie pakalpojumi, limits EUR"] = (
        _amount_in_tables_by_row(parsed, r"Maksas\s+stacionār(ie|ie)\s+pakalpojumi", min_v=1, max_v=1_000_000) or
        _amount_near_kw(raw_text, "Maksas stacionārie pakalpojumi", max_v=1_000_000) or "-"
    )

    feats["Maksas stacionārā rehabilitācija, limits EUR"] = (
        _amount_in_tables_by_row(parsed, r"stacionār[āa]\s+rehabilitāc", min_v=1, max_v=1_000_000) or
        _amount_near_kw(raw_text, "Maksas stacionārā rehabilitācija", max_v=1_000_000) or "-"
    )

    feats["Ambulatorā rehabilitācija"] = (
        _amount_in_tables_by_row(parsed, r"Ambulator[āa]\s+rehabilitāc", min_v=1, max_v=1_000_000) or
        _amount_near_kw(raw_text, "Ambulatorā rehabilitācija", max_v=1_000_000) or "-"
    )

    feats["Piemaksa par plastikāta kartēm, EUR"] = (
        _amount_in_tables_by_row(parsed, r"Piemaksa\s+par\s+plastikāta\s+kartēm", min_v=1, max_v=100) or
        _amount_near_kw(raw_text, "Piemaksa par plastikāta kartēm", max_v=100) or "-"
    )

    feats["Zobārstniecība ar 50% atlaidi (pamatpolise)"] = (
        _amount_in_tables_by_row(parsed, r"Zobārstniecība\s+ar\s+50%\s+atlaidi.*pamat", min_v=1, max_v=10_000) or "-"
    )

    feats["Zobārstniecība ar 50% atlaidi, apdrošinājuma summa (pp)"] = (
        _amount_in_tables_by_row(parsed, r"Zobārstniecība\s+ar\s+50%\s+atlaidi", min_v=1, max_v=10_000) or "-"
    )

    feats["Vakcinācija pret ērčiem un gripu"] = (
        _amount_in_tables_by_row(parsed, r"Profilaktisk[āa]\s+vakcin[āa]cija", min_v=1, max_v=10_000) or
        feats.get("Vakcinācija, limits EUR", "-")
    )

    feats["Ambulatorā rehabilitācija (pp)"] = (
        _amount_in_tables_by_row(parsed, r"Ambulator[āa]\s+rehabilitāc", min_v=1, max_v=1_000_000) or "-"
    )

    feats["Medikamenti ar 50% atlaidi"] = (
        _amount_in_tables_by_row(parsed, r"Medikamenti\s+ar\s+50%\s+atlaidi", min_v=1, max_v=50_000) or "-"
    )

    feats["Sports"] = (
        _amount_in_tables_by_row(parsed, r"\bSports\b", min_v=1, max_v=50_000) or
        ("v" if _text_has_kw(raw_text, "Sports") else "-")
    )

    crit_amt = (
        _amount_in_tables_by_row(parsed, r"Kritisk[āa]s\s+saslim", min_v=1, max_v=1_000_000) or
        _amount_near_kw(raw_text, "Kritiskās saslimšanas", max_v=1_000_000)
    )
    feats["Kritiskās saslimšanas"] = (f"{int(crit_amt)}" if crit_amt else "ir iekļauts") if (
        _text_has_kw(raw_text, "Kritiskās saslimšanas")
    ) else "-"

    feats["Maksas stacionārie pakalpojumi, limits EUR (pp)"] = (
        "ir iekļauts" if _text_has_kw(raw_text, "Maksas stacionārie pakalpojumi") else "-"
    )

    prog = ProgramModel(
        insurer="COM_VA",
        program_code=program_code_text,
        base_sum_eur=float(base_sum or 0.0),
        premium_eur=float(premium or 0.0),
        payment_method="-",
        features=normalize_features(feats, "COM_VA", program_code_text, float(premium or 0.0)),
    )
    return [prog]


# ----------------- BTA table scraping (header + row fallback) -----------------
def _find_program_table_layout(tbl: List[List[str]]) -> Optional[Dict[str, int]]:
    for r_idx, row in enumerate(tbl):
        row_norm = [_norm_text(c) for c in row]
        if not any(row_norm):
            continue
        c_program = _find_col_idx(row_norm, ["programma"])
        c_sum     = _find_col_idx(row_norm, ["apdrošinājuma summa"])
        c_prem    = _find_col_idx(row_norm, ["prēmija 1 personai", "prēmija 1 personai (eur)", "prēmija"])
        if c_program is not None and c_sum is not None and c_prem is not None:
            return {"header_row": r_idx, "c_program": c_program, "c_sum": c_sum, "c_prem": c_prem}
    return None

def _row_has_program_name(row: List[str]) -> Optional[str]:
    text = " ".join(_norm_text(c) for c in row)
    tok = _find_prog_token(text)
    return tok

def _row_extract_two_numbers(row_and_neighbors: List[List[str]]) -> Optional[tuple]:
    """
    Pull plausible (base_sum, premium) from a row + next two neighbors.
    """
    joined = " ".join(" ".join(_norm_text(c) for c in r) for r in row_and_neighbors if r)
    nums = _extract_numbers_no_percent(joined)
    if not nums:
        return None

    big = [n for n in nums if n >= 1000]
    if not big:
        big = [n for n in nums if n >= 500]
    if not big:
        return None
    base_sum = max(big)

    prem_cands = [n for n in nums if 50 <= n <= 5000 and n < base_sum]
    if not prem_cands:
        return None

    premium = max(prem_cands)
    return (base_sum, premium)

def _row_max_premium_candidate(row: List[str], base_sum: Optional[float]) -> Optional[float]:
    txt = " ".join(_norm_text(c) for c in row)
    nums = _extract_numbers_no_percent(txt)
    if base_sum is not None:
        nums = [n for n in nums if n < base_sum]
    cand = [n for n in nums if 50 <= n <= 5000]
    return max(cand) if cand else None

def _rescue_premiums_from_tables(parsed: Dict[str, Any], progs: List[ProgramModel]) -> List[ProgramModel]:
    """
    If any program has premium <= 0, try to fill it from tables.
    Also upgrades base_sum when a larger value is found, and patches payment_method.
    """
    if not progs:
        return progs

    tables = parsed.get("tables") or []
    if not tables:
        return progs

    # -------- Phase 1: with detected layout --------
    best_tbl = None
    best_layout = None
    best_score = 0
    for tbl in tables:
        layout, score = _find_program_table_layout_scored(tbl)
        if layout and score > best_score:
            best_tbl, best_layout, best_score = tbl, layout, score

    if best_tbl and best_layout:
        header_row = best_layout["header_row"]
        c_sum = best_layout["c_sum"]

        def _canon_name(s: str) -> str:
            return _norm_text(s).lower().replace(" ", "")

        need = { _canon_name(p.program_code): p for p in progs if (not p.premium_eur) or (p.premium_eur <= 0) }
        for row in best_tbl[header_row + 1:]:
            if not need or not row:
                continue
            rtxt = " ".join(_norm_text(x) for x in row)
            pname = _find_prog_token(rtxt)
            key = _canon_name(pname or "")
            if key and key in need:
                base_sum = _num_safe(row[c_sum]) if c_sum < len(row) else None
                prem = _row_max_premium_candidate(row, base_sum)
                if prem is not None and 50 <= prem <= 5000 and (base_sum is None or prem < base_sum):
                    p = need[key]
                    if base_sum and (p.base_sum_eur < 1000 or base_sum > p.base_sum_eur):
                        p.base_sum_eur = float(base_sum)
                        if "Apdrošinājuma summa pamatpolisei, EUR" in p.features:
                            p.features["Apdrošinājuma summa pamatpolisei, EUR"] = p.base_sum_eur
                    p.premium_eur = float(prem)
                    if "Pamatpolises prēmija 1 darbiniekam, EUR" in p.features:
                        p.features["Pamatpolises prēmija 1 darbiniekam, EUR"] = p.premium_eur
                    need.pop(key, None)

    # -------- Phase 2: headerless fallback --------
    import unicodedata
    def _canon_all(s: str) -> str:
        t = _norm_text(s)
        t = unicodedata.normalize("NFD", t)
        t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
        return t.lower().replace(" ", "")

    for p in progs:
        if p.premium_eur and p.premium_eur > 0:
            continue

        found_base = None
        found_prem = None

        for tbl in tables:
            for i, row in enumerate(tbl or []):
                row_txt = " ".join(_norm_text(x) for x in row)
                row_key = _canon_all(row_txt)
                anchor = _canon_all(p.program_code)
                if anchor not in row_key and anchor != _canon_all("Pamatprogramma"):
                    continue

                neighbors = [row]
                if i + 1 < len(tbl): neighbors.append(tbl[i + 1])
                if i + 2 < len(tbl): neighbors.append(tbl[i + 2])

                pair = _row_extract_two_numbers(neighbors)
                if pair:
                    found_base, found_prem = pair
                else:
                    nums = _extract_numbers_no_percent(
                        " ".join(" ".join(_norm_text(c) for c in r) for r in neighbors if r)
                    )
                    base_big = [n for n in nums if n >= 1000]
                    found_base = max(base_big) if base_big else None
                    cands = [n for n in nums if 50 <= n <= 5000 and (found_base is None or n < found_base)]
                    found_prem = max(cands) if cands else None

                if found_prem is not None and 50 <= found_prem <= 5000:
                    break
            if found_prem is not None:
                break

        if found_base and (p.base_sum_eur < 1000 or found_base > p.base_sum_eur):
            p.base_sum_eur = float(found_base)
            if "Apdrošinājuma summa pamatpolisei, EUR" in p.features:
                p.features["Apdrošinājuma summa pamatpolisei, EUR"] = p.base_sum_eur

        if found_prem is not None:
            p.premium_eur = float(found_prem)
            if "Pamatpolises prēmija 1 darbiniekam, EUR" in p.features:
                p.features["Pamatpolises prēmija 1 darbiniekam, EUR"] = p.premium_eur

    # Patch payment method if it’s still '-'
    pay = _extract_payment_terms(parsed)
    if pay:
        for p in progs:
            if not p.payment_method or p.payment_method.strip() == "-":
                p.payment_method = pay
                if "Pakalpojuma apmaksas veids" in p.features:
                    p.features["Pakalpojuma apmaksas veids"] = pay

    return progs

def bta_program_table(parsed: Dict[str, Any]) -> List[ProgramModel]:
    """
    Extract ONLY main rows (V2+, V3+, Pamatprogramma) from the MAIN program table.
    Prefer the premium column for 'Visiem (52)' and then re-select the premium
    as the LARGEST plausible number (50..5000) < base_sum present in the row.
    """
    import unicodedata

    def _canon(s: str) -> str:
        t = _norm_text(s)
        t = unicodedata.normalize("NFD", t)
        t = "".join(c for c in t if unicodedata.category(c) != "Mn")
        return t.lower()

    def _canon_ns(s: str) -> str:
        return _canon(s).replace(" ", "")

    tables = parsed.get("tables") or []
    raw_text = parsed.get("raw_text") or ""
    addons = _collect_bta_addons(parsed)
    pacienta_iemaksa = _extract_pacienta_iemaksa(parsed)
    payment_terms = _extract_payment_terms(parsed)
    programs: List[ProgramModel] = []

    # ---------- choose the SINGLE best candidate table ----------
    best_tbl = None
    best_layout = None
    best_score = 0
    for tbl in tables:
        layout, score = _find_program_table_layout_scored(tbl)
        if layout and score > best_score:
            best_tbl = tbl
            best_layout = layout
            best_score = score

    def _premium_candidates_in_header(tbl: List[List[str]], header_row: int) -> List[int]:
        out: List[int] = []
        if not tbl or header_row < 0 or header_row >= len(tbl):
            return out
        top = max(0, header_row - 2)
        bot = min(len(tbl) - 1, header_row + 2)
        max_cols = max(len(r) for r in tbl[top:bot+1]) if tbl else 0
        for c in range(max_cols):
            parts: List[str] = []
            for rr in range(top, bot + 1):
                if c < len(tbl[rr]):
                    parts.append(_norm_text(tbl[rr][c]))
            h = " ".join(parts)
            h_norm = _canon(h)
            h_ns = _canon_ns(h)
            looks_prem = (
                ("premij" in h_norm or "premij" in h_ns or "prem" in h_ns) and
                ("person" in h_norm or "person" in h_ns or "(eur)" in h_norm or "eur" in h_norm)
            )
            if looks_prem:
                out.append(c)
        return out

    def _refine_to_visiem(tbl: List[List[str]], header_row: int, prem_idxs: List[int]) -> Optional[int]:
        if not prem_idxs:
            return None
        def stacked_header(c: int) -> str:
            parts: List[str] = []
            for rr in range(max(0, header_row - 2), min(len(tbl), header_row + 3)):
                if c < len(tbl[rr]):
                    parts.append(_norm_text(tbl[rr][c]))
            return _canon(" ".join(parts))
        best_c = None
        best_sc = -1
        for c in prem_idxs:
            h = stacked_header(c)
            score = 0
            if "visiem" in h: score += 1000
            if "(52" in h or "52)" in h or "( 52" in h or "52 )" in h: score += 1000
            if " 52 " in h or "52 " in h or " 52" in h: score += 200
            score += int(c * 0.1)
            if score > best_sc:
                best_sc, best_c = score, c
        return best_c if best_sc > 0 else None

    def _emit_row(pname: str, base_sum: float, premium: float):
        display_name = "Pamatprogramma" if "PAMATPROGRAMMA" in pname.upper() else pname
        code = display_name
        feats = {
            "Programmas nosaukums": display_name,
            "Pakalpojuma apmaksas veids": payment_terms or "-",
            "Apdrošinājuma summa pamatpolisei, EUR": base_sum,
            "Pamatpolises prēmija 1 darbiniekam, EUR": premium,
        }
        if "Pacientu iemaksa" in LV_FEATURE_KEYS and pacienta_iemaksa:
            feats["Pacientu iemaksa"] = pacienta_iemaksa

        prog = ProgramModel(
            insurer="BTA",
            program_code=code,
            base_sum_eur=float(base_sum),
            premium_eur=float(premium),
            payment_method=payment_terms or "-",
            features=feats,
        )
        prog.features = normalize_features(prog.features, "BTA", prog.program_code, prog.premium_eur)
        prog.features = fill_bta_detail_fields(prog.features, parsed)
        prog.features = _apply_bta_bottom_rules(prog.features, addons)
        programs.append(prog)

    # ---------- read rows from the chosen table ----------
    if best_tbl and best_layout:
        header_row = best_layout["header_row"]
        c_program  = best_layout["c_program"]
        c_sum      = best_layout["c_sum"]
        c_prem     = best_layout["c_prem"]

        prem_idxs = _premium_candidates_in_header(best_tbl, header_row)
        prefer = _refine_to_visiem(best_tbl, header_row, prem_idxs)
        if prefer is not None:
            c_prem = prefer

        if os.getenv("DEBUG_BTA"):
            try:
                hdr_txt = _norm_text(best_tbl[header_row][c_prem])
            except Exception:
                hdr_txt = "(n/a)"
            print("BTA chosen premium header:", hdr_txt)

        start = header_row + 1
        for row in best_tbl[start:]:
            if not row or all(_norm_text(c) == "" for c in row):
                continue

            pname_cell = _norm_text(row[c_program]) if c_program < len(row) else ""
            rtxt = " ".join(_norm_text(x) for x in row)

            # robust program detection (keeps '+')
            pname = _find_prog_token(pname_cell) or _find_prog_token(rtxt)
            if not pname:
                continue

            base_sum = _num_safe(row[c_sum]) if c_sum < len(row) else None
            premium = _num_safe(row[c_prem]) if c_prem < len(row) else None

            # ALWAYS re-select premium from the row: largest plausible premium < base_sum
            alt = _row_max_premium_candidate(row, base_sum)
            if alt is not None:
                premium = alt

            # upgrade either side if pair available
            if (base_sum is None or base_sum < 1000) or (premium is None or premium < 50 or premium > 5000):
                pair = _row_extract_two_numbers([row])
                if pair:
                    b2, p2 = pair
                    if (base_sum is None or base_sum < 1000) and b2:
                        base_sum = b2
                    if (premium is None or premium < 50 or premium > 5000) and p2:
                        premium = p2

            if os.getenv("DEBUG_BTA"):
                print("[BTA]", pname, "base=", base_sum, "premium=", premium,
                      "nums=", _extract_numbers_no_percent(" ".join(_norm_text(x) for x in row)))

            # ONLY emit when premium is actually valid (no defaulting to 0)
            if base_sum and premium and 50 <= premium <= 5000 and (base_sum is None or premium < base_sum):
                _emit_row(pname, base_sum, premium)

    # ---------- last resort: global row-scan fallback ----------
    if not programs:
        for tbl in tables:
            for i, row in enumerate(tbl or []):
                pname = _row_has_program_name(row)
                if not pname:
                    continue
                neighbors = [row]
                if i + 1 < len(tbl): neighbors.append(tbl[i + 1])
                if i + 2 < len(tbl): neighbors.append(tbl[i + 2])
                pair = _row_extract_two_numbers(neighbors)
                if not pair:
                    base_sum = _value_from_text_near(raw_text, pname, r"Apdrošinājuma\s+summa", 1000, 10_000_000, 900)
                    premium  = _value_from_text_near(raw_text,  pname, r"Prēmija",               50,        5000, 900)
                    if not base_sum or not premium:
                        continue
                else:
                    base_sum, premium = pair
                _emit_row("Pamatprogramma" if pname.lower().startswith("pamat") else pname, base_sum, premium)

    # de-dup by program_code (keep the one with bigger base_sum if duplicates)
    dedup: Dict[str, ProgramModel] = {}
    for p in programs:
        key = p.program_code.upper()
        if key not in dedup or p.base_sum_eur > dedup[key].base_sum_eur:
            dedup[key] = p

    out = list(dedup.values())
    return out

# ----------------- BTA raw-text fallback (tightened) -----------------
def bta_programs_from_text(parsed: Dict[str, Any]) -> List[ProgramModel]:
    """
    Brochure/text fallback:
      - base_sum: from an explicit 'Apdrošinājuma summa' near the program anchor
      - premium: ONLY from an explicit 'Prēmija' label; if absent → 0 (don’t guess from other numbers)
    """
    raw = parsed.get("raw_text") or ""
    if not raw:
        return []

    addons = _collect_bta_addons(parsed)
    pacienta_iemaksa = _extract_pacienta_iemaksa(parsed)
    payment_terms = _extract_payment_terms(parsed)
    programs: List[ProgramModel] = []

    # scan tokens; regex captures '+' when present
    for m in re.finditer(r"(V\d+\+?|Pamatprogramma)", raw, re.IGNORECASE):
        pname = m.group(1)
        display_name = "Pamatprogramma" if pname.lower().startswith("pamat") else pname

        start = max(0, m.start() - 800)
        end   = min(len(raw), m.end() + 800)
        window = raw[start:end]

        ms = re.search(r"Apdrošinājuma\s+summa[^\d%]{0,60}([0-9][0-9 \u00A0\.,]+)", window, re.IGNORECASE)
        base_sum = _num_safe(ms.group(1)) if ms else None

        mp = re.search(r"Prēmija(?:\s*1\s*personai[^\d%]{0,60})?([0-9][0-9 \u00A0\.,]+)", window, re.IGNORECASE)
        premium = _num_safe(mp.group(1)) if mp else 0.0

        if not base_sum:
            nums = _extract_numbers_no_percent(window)
            big = [n for n in nums if n >= 1000 and float(n).is_integer()]
            base_sum = max(big) if big else None

        if not base_sum:
            continue

        code = display_name
        features = {
            "Programmas nosaukums": display_name,
            "Pakalpojuma apmaksas veids": payment_terms or "-",
            "Apdrošinājuma summa pamatpolisei, EUR": base_sum,
            "Pamatpolises prēmija 1 darbiniekam, EUR": premium,
        }
        if "Pacientu iemaksa" in LV_FEATURE_KEYS and pacienta_iemaksa:
            features["Pacientu iemaksa"] = pacienta_iemaksa

        prog = ProgramModel(
            insurer="BTA",
            program_code=code,
            base_sum_eur=float(base_sum),
            premium_eur=float(premium),
            payment_method=payment_terms or "-",
            features=features,
        )
        prog.features = normalize_features(prog.features, "BTA", prog.program_code, prog.premium_eur)
        prog.features = fill_bta_detail_fields(prog.features, parsed)
        prog.features = _apply_bta_bottom_rules(prog.features, addons)
        programs.append(prog)

    # de-dup
    dedup: Dict[str, ProgramModel] = {}
    for p in programs:
        key = p.program_code.upper()
        if key not in dedup or p.base_sum_eur > dedup[key].base_sum_eur:
            dedup[key] = p

    out = list(dedup.values())
    return out

# -------- BTA flavor scoring & routing --------
def _score_bta_offer_grid(parsed: Dict[str, Any]) -> int:
    """
    Score if the document looks like the short 'offer grid' (tables with Programma / Summa / Prēmija columns).
    """
    tables = parsed.get("tables") or []
    if not tables:
        return 0

    best_layout_score = 0
    program_row_hits = 0
    premium_hits = 0

    for tbl in tables:
        layout, score = _find_program_table_layout_scored(tbl)
        best_layout_score = max(best_layout_score, score or 0)
        if not layout:
            continue

        header_row = layout["header_row"]
        c_sum = layout["c_sum"]
        # scan a few rows after header for program tokens + plausible numbers
        for rr in range(header_row + 1, min(len(tbl), header_row + 18)):
            row = tbl[rr]
            rtxt = " ".join(_norm_text(x) for x in (row or []))
            if _find_prog_token(rtxt):
                program_row_hits += 1
                nums = _extract_numbers_no_percent(rtxt)
                base_big = any(n >= 1000 for n in nums)
                prem_ok = any(50 <= n <= 5000 for n in nums)
                if base_big and prem_ok:
                    premium_hits += 1

    # heuristic scoring
    return int(best_layout_score) + program_row_hits * 30 + premium_hits * 40


def _score_bta_brochure(parsed: Dict[str, Any]) -> int:
    """
    Score if the document looks like the long brochure text (Programmas apraksts, BTA apmaksā, bullets).
    """
    raw = (_norm_text(parsed.get("raw_text")) or "").upper()
    if not raw:
        return 0

    kw = [
        "PROGRAMMAS APR",      # PROGRAMMAS APRAKSTS / APR.
        "BTA APMAKSĀ",
        "BTA NEAPMAKSĀ",
        "PROGRAMMA – ZOBĀRSTNIECĪBA",
        "PROGRAMMA- ZOBĀRSTNIECĪBA",
        "PROGRAMMA ZOBĀRSTNIECĪBA",
    ]
    score = 0
    for k in kw:
        if k in raw:
            score += 150

    # bullet density
    bullets = raw.count("•") + raw.count("▪") + raw.count("●") + raw.count("►")
    if bullets >= 5:
        score += 80 + min(200, bullets * 5)

    # penalize if explicit grid header words are present (rare in brochure)
    if "PRĒMIJA 1 PERSONAI" in raw or "PROGRAMMA" in raw and "APDROŠINĀJUMA SUMMA" in raw:
        score -= 60

    return score


def _choose_bta_path(hint: str, parsed: Dict[str, Any]) -> str:
    """
    Decide 'table' vs 'text' for BTA based on explicit hint and a light heuristic.
    Returns 'table' or 'text'.
    """
    h = (hint or "").strip().lower()
    if h in ("bta2", "bta:brochure", "bta-brochure", "bta_text", "bta-text"):
        return "text"
    if h in ("bta", "bta:offer", "bta_offer", "bta:auto", "bta-auto"):
        # auto: prefer table unless brochure clearly wins
        offer = _score_bta_offer_grid(parsed)
        brochure = _score_bta_brochure(parsed)
        return "text" if brochure > offer else "table"
    if "bta" in h:
        return "table"
    return "table"

# ----------------- Non-BTA (AI) + BTA router -----------------
async def ai_enrich_and_validate(parsed: Dict[str, Any], company_hint: str | None) -> List[ProgramModel]:
    hint_raw = (company_hint or "").strip()
    hint = hint_raw.lower()
    
# ----- COM / Compensa path (run before BTA) -----
    mapped = _company_hint_to_code(company_hint or "")
    if mapped == "COM_VA":
        progs = com_programs(parsed)
        if progs:
            return progs
# if COM-specific path failed, continue to the rest

    if "bta" in hint:
        path = _choose_bta_path(hint, parsed)

        if path == "table":
            progs: List[ProgramModel] = bta_program_table(parsed)
            if not progs:
                offer = _score_bta_offer_grid(parsed)
                brochure = _score_bta_brochure(parsed)
                if brochure >= offer:
                    progs = bta_programs_from_text(parsed)
            if not progs:
                progs = bta_programs_from_text(parsed)
        else:
            progs = bta_programs_from_text(parsed)

        # rescue premiums from the tables EVEN IF we came from text
        if progs and any((p.premium_eur is None) or (p.premium_eur <= 0) for p in progs):
            progs = _rescue_premiums_from_tables(parsed, progs)

        # >>> BTA2: force SINGLE program (no filename reliance) <<<
        if hint.startswith("bta2") and progs and len(progs) > 1:
            progs = _force_single_for_bta2(progs, parsed)

        # >>> BTA2: PREMIUM MASK — do NOT alter other logic <<<
        if hint.startswith("bta2") and progs:
            for p in progs:
                # keep numeric field safe for the schema, but mask feature display as "-"
                if "Pamatpolises prēmija 1 darbiniekam, EUR" in p.features:
                    p.features["Pamatpolises prēmija 1 darbiniekam, EUR"] = "-"
                # if someone mirrored a premium into features["premium_eur"], mask that too
                if "premium_eur" in p.features:
                    p.features["premium_eur"] = "-"
                # optionally zero the numeric value to avoid mismatched downstream math
                p.premium_eur = 0.0

        if progs:
            return progs

        # If BTA routing produced nothing, fall through to generic AI or naive fallback below.

    # ---------- Non-BTA or fallback path ----------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback_naive(parsed, company_hint)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = build_prompt(parsed, company_hint)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"Return ONLY strict JSON (no prose)."},
                {"role":"system","content":SYSTEM},
                {"role":"user","content":prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        js = _extract_json_from_text(content) or "{}"
        data = json.loads(js)
        extracted = ExtractionModel.model_validate(data)
        programs = extracted.programs

        if company_hint:
            # >>> STRICT insurer normalization for non-BTA <<<
            normalized_insurer = _company_hint_to_code(company_hint) or company_hint
            for p in programs:
                p.insurer = normalized_insurer
                p.features = normalize_features(
                    p.features if isinstance(p.features, dict) else {},
                    insurer=normalized_insurer,
                    program_code=p.program_code,
                    premium_eur=p.premium_eur,
                )

        # BTA2 singleton (even for LLM path)
        if hint.startswith("bta2") and programs and len(programs) > 1:
            programs = _force_single_for_bta2(programs, parsed)

        # >>> BTA2: PREMIUM MASK on LLM path as well <<<
        if hint.startswith("bta2") and programs:
            for p in programs:
                if "Pamatpolises prēmija 1 darbiniekam, EUR" in p.features:
                    p.features["Pamatpolises prēmija 1 darbiniekam, EUR"] = "-"
                if "premium_eur" in p.features:
                    p.features["premium_eur"] = "-"
                p.premium_eur = 0.0

        return programs

    except Exception:
        return fallback_naive(parsed, company_hint)


# ----------------- Naive fallback -----------------
def fallback_naive(parsed: Dict[str, Any], company_hint: str | None) -> List[ProgramModel]:
    text = parsed.get("raw_text") or ""
    raw_hint = (company_hint or "").strip()
    insurer_hint = raw_hint.lower()
    # normalize with STRICT mapping (BTA/BTA2 → 'BTA'; others to fixed codes)
    insurer = _company_hint_to_code(raw_hint) or (raw_hint or "Nezināms")

    program_code = "PROGRAMMA_1"
    if "VARIANTS" in text.upper(): program_code = "V1 PLUSS C10"
    elif "Pamatprogramma" in text: program_code = "Dzintara polise pluss 2"
    elif "POLISE" in text.upper(): program_code = "POLISE I"

    features = {k: "-" for k in LV_FEATURE_KEYS}
    if "Apdrošinātājs" in features: features["Apdrošinātājs"] = insurer
    if "Programmas kods" in features: features["Programmas kods"] = program_code
    if "Pakalpojuma apmaksas veids" in features: features["Pakalpojuma apmaksas veids"] = "pēc cenrāža"
    if "Apdrošinājuma summa pamatpolisei, EUR" in features: features["Apdrošinājuma summa pamatpolisei, EUR"] = 0
    if "Pamatpolises prēmija 1 darbiniekam, EUR" in features: features["Pamatpolises prēmija 1 darbiniekam, EUR"] = 0

    return [ProgramModel(
        insurer=insurer,
        program_code=program_code,
        base_sum_eur=0.0,
        premium_eur=0.0,
        payment_method="pēc cenrāža",
        features=features,
    )]
