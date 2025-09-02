LV_FEATURE_KEYS = [
    "Apdrošinātājs",
    "Programmas nosaukums",
    "Programmas kods",
    "Pakalpojuma apmaksas veids",
    "Apdrošinājuma summa pamatpolisei, EUR",
    "Pacientu iemaksa",
    "Maksas ģimenes ārsta mājas vizītes, limits EUR",
    "Maksas ģimenes ārsta, internista, terapeita un pediatra konsultācija, limits EUR",
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

# JSON Schema for OpenAI Responses API structured output
def json_schema():
    # Allow ONLY the 25 Latvian keys; no extra keys
    features_props = {k: {"type": ["string", "number"]} for k in LV_FEATURE_KEYS}
    return {
        "name": "InsuranceExtraction",
        "schema": {
            "type": "object",
            "properties": {
                "programs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "insurer": {"type": "string"},
                            "program_code": {"type": "string"},
                            "base_sum_eur": {"type": "number"},
                            "premium_eur": {"type": "number"},
                            "payment_method": {"type": ["string", "null"]},
                            "features": {
                                "type": "object",
                                "properties": features_props,
                                "required": LV_FEATURE_KEYS,
                                "additionalProperties": False
                            },
                        },
                        "required": [
                            "insurer", "program_code",
                            "base_sum_eur", "premium_eur", "features"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["programs"],
            "additionalProperties": False
        }
    }

