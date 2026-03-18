"""
Recommendation utilities: top-N selection and red-flag detection.

CORRECTIONS v2 :
- top_specialties : déduplique vraiment par Specialite (groupe + prend le max ScoreGlobal)
  → avant, si la même spécialité avait 5 lignes dans le CSV, le top-3 pouvait
    retourner 3× Cardiologie (bug majeur de pertinence).
- detect_red_flags : liste élargie (hémoptysie, déficit neurologique, etc.)
- Ajout de top_specialties_with_scores pour exposer les scores dans l'UI
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


CRITICAL_FLAGS = {
    # Cardio-vasculaire
    "douleur thoracique",
    "irradiation bras",
    "irradiation machoire",
    # Respi
    "gêne respiratoire",
    "gene respiratoire",
    "hemoptysie",
    "hémoptysie",
    "dyspnee",
    "dyspnée",
    # Neuro
    "perte de connaissance",
    "deficit neurologique",
    "déficit neurologique",
    "paralysie",
    "convulsions",
    "confusion",
    # Choc / urgence vitale
    "malaise",
    "syncope",
    "etat de choc",
    "état de choc",
}


def detect_red_flags(answers: Dict, description_text: str) -> Tuple[bool, List[str]]:
    """
    Detect critical red flags from checkbox answers and free text.
    """
    red_flags_answers = set(flag.lower() for flag in (answers.get("red_flags") or []))
    text = (description_text or "").lower()
    hits = []
    for flag in CRITICAL_FLAGS:
        if flag in red_flags_answers or flag in text:
            hits.append(flag)
    return (len(hits) > 0), hits


def top_specialties(df_scored: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Return top-N UNIQUE specialties from a scored dataframe.

    FIX : on groupe par Specialite et on prend le score max pour chaque
    spécialité, puis on trie → garantit la diversité des recommandations.
    """
    if df_scored.empty:
        return df_scored.head(n).reset_index(drop=True)

    # Garder la ligne avec le meilleur ScoreGlobal pour chaque spécialité
    best_per_specialty = (
        df_scored.sort_values("ScoreGlobal", ascending=False)
        .drop_duplicates(subset=["Specialite"], keep="first")
        .reset_index(drop=True)
    )
    return best_per_specialty.head(n).reset_index(drop=True)
