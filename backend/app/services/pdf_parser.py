import re
from pathlib import Path

from pypdf import PdfReader

from app.feature_schema import FEATURE_SCHEMA, REQUIRED_INPUT_KEYS
from app.schemas import ParsedField


VALUE_PATTERNS: dict[str, list[str]] = {
    "fasting_glucose_mmol_l": [
        r"(?:fasting\s+glucose|glucose|глюкоза)[^\d]{0,25}(\d+[.,]?\d*)\s*(?:mmol/?l|ммоль/?л)",
    ],
    "fasting_glucose_mg_dl": [
        r"(?:fasting\s+glucose|glucose)[^\d]{0,25}(\d+[.,]?\d*)\s*(?:mg/?dl)",
    ],
    "insulin": [
        r"(?:insulin|инсулин)[^\d]{0,20}(\d+[.,]?\d*)",
    ],
    "homa_ir": [
        r"(?:homa[\s\-]?ir|хома[\s\-]?ir)[^\d]{0,20}(\d+[.,]?\d*)",
    ],
    "hba1c": [
        r"(?:hba1c|glycated\s+hemoglobin|гликированный\s+гемоглобин)[^\d]{0,20}(\d+[.,]?\d*)",
    ],
    "cholesterol_total": [
        r"(?:total\s+cholesterol|cholesterol|холестерин)[^\d]{0,20}(\d+[.,]?\d*)",
    ],
}


def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        return None


def _field_label(key: str) -> str:
    for field in FEATURE_SCHEMA:
        if field["key"] == key:
            return field["label"]
    return key


def parse_lab_report_text(text: str) -> list[ParsedField]:
    extracted: list[ParsedField] = []
    lowered = text.lower()
    for key, patterns in VALUE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE | re.MULTILINE)
            if not match:
                continue
            value = _to_float(match.group(1))
            if value is None:
                continue
            extracted.append(
                ParsedField(
                    key=key,
                    label=_field_label(key),
                    value=value,
                    confidence=0.9 if key in {"fasting_glucose_mmol_l", "insulin", "homa_ir"} else 0.82,
                    source_text=match.group(0).strip(),
                    required_for_prediction=key in REQUIRED_INPUT_KEYS,
                )
            )
            break
    return extracted
