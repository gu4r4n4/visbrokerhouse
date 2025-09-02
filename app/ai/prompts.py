from textwrap import dedent

SYSTEM = """You are a precise extraction engine for insurance offers in Latvian market.
Return ONLY valid JSON per schema. Do not omit any program. If a document has multiple sections/tables, include all programs.
All feature keys MUST be in Latvian as provided (25 rows schema). Numeric fields must be numbers, not strings. Currency is EUR.
"""

def build_prompt(parsed: dict, company_hint: str | None) -> str:
    # Keep the prompt compact; include important slices of text and first tables
    head = f"Company hint: {company_hint or 'unknown'}."
    raw_excerpt = (parsed.get("raw_text") or "")[:12000]  # clamp to avoid over-long prompts
    return dedent(f"""
    Extract ALL insurance programs from the document.
    Rules by company (Compensa, Balta, Gjensidige, Seesam) apply as follows (summary):
    - Compensa: program names under 'VARIANTS' tables (first row); take 'Apdrošinājuma summa vienai personai, EUR' and 'Premija vienai personai, EUR' per program.
    - Balta: program name = concat 'Ambulatorā un stacionārā palīdzība' (e.g., AS1), + 'Laboratoriskie izmeklējumi' (e.g., LAB_1/2), + 'Stacionārie maksas pakalpojumi' (amount only, e.g., 250).
    - Gjensidige: table 'Pamatprogramma'; first row has program names; take 'Apdrošinājuma summa 1 personai gadā, EUR' and 'Prēmija 1 personai gadā, EUR'.
    - Seesam: sections 'POLISE I', 'POLISE II.I', etc.; take 'Apdrošinājuma summa vienai personai, EUR' and 'Premija vienai personai, EUR'.

    For EVERY program, fill the 25 feature rows (Latvian keys). If data absent, write '-' (dash) or sensible textual markers like 'bez gada apakšlimita' where appropriate.
    Return JSON strictly matching the schema (fields: insurer, program_code, base_sum_eur, premium_eur, payment_method, features{{25 keys}}).

    Document raw text excerpt (may be truncated):
    ---
    {raw_excerpt}
    ---
    """).strip()
