# app/evidence.py
from __future__ import annotations
from typing import List, Optional

_CANON = {
    "Randomized Controlled Trial": "RCT",
    "Clinical Trial": "Clinical trial",
    "Meta-Analysis": "Meta-analysis",
    "Systematic Review": "Systematic review",
    "Review": "Review",
    "Cohort Studies": "Cohort",
    "Case-Control Studies": "Case-control",
    "Cross-Sectional Studies": "Cross-sectional",
    "Comparative Study": "Comparative",
    "Observational Study": "Observational",
    "Multicenter Study": "Multicenter",
    "Letter": "Letter",
    "Editorial": "Editorial",
}

# Prefer high-signal sections when ranking PMC full text
SECTION_WEIGHTS = {
    "Results": 1.20,
    "Methods": 1.10,
    "Discussion": 1.05,
    "Conclusion": 1.05,
    "Limitations": 1.05,
}

def classify_study_type(pubtypes: List[str], title: str = "") -> str:
    """
    Map PubMed pubtypes (and sometimes title hints) to a concise evidence label.
    Picks the highest-priority match when multiple types are present.
    """
    priority = ["RCT", "Meta-analysis", "Systematic review", "Cohort", "Case-control",
                "Cross-sectional", "Clinical trial", "Observational", "Comparative",
                "Multicenter", "Review", "Editorial", "Letter"]
    found = set()
    for pt in pubtypes or []:
        if pt in _CANON:
            found.add(_CANON[pt])
    tl = (title or "").lower()
    if "systematic review" in tl:
        found.add("Systematic review")
    if "meta-analysis" in tl or "meta analysis" in tl:
        found.add("Meta-analysis")
    for tag in priority:
        if tag in found:
            return tag
    return "Unspecified"

def preference_boost(study_type: str, prefer_types: List[str] | None) -> float:
    """
    Multiplicative boost for preferred study types (e.g., RCT, Meta-analysis).
    """
    if not prefer_types:
        return 1.0
    norm = {t.strip().lower() for t in prefer_types}
    return 1.2 if study_type.lower() in norm else 1.0

def section_boost(section: str) -> float:
    """
    Multiplicative boost for desired PMC sections (e.g., Results, Methods).
    """
    return SECTION_WEIGHTS.get(section, 1.0)
