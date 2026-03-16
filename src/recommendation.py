"""
Recommendation utilities: top-N selection and red-flag detection.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


CRITICAL_FLAGS = {
    "douleur thoracique",
    "perte de connaissance",
    "gêne respiratoire",
    "gene respiratoire",
    "hemoptysie",
}


def detect_red_flags(answers: Dict, description_text: str) -> Tuple[bool, List[str]]:
    """
    Detect critical red flags from checkbox answers and free text.
    """
    red_flags = set(flag.lower() for flag in (answers.get("red_flags") or []))
    text = (description_text or "").lower()
    hits = []
    for flag in CRITICAL_FLAGS:
        if flag in red_flags or flag in text:
            hits.append(flag)
    return (len(hits) > 0), hits


def top_specialties(df_scored: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Return top-N specialties from a scored dataframe.
    """
    return df_scored.head(n).reset_index(drop=True)
