"""
Score computation: combine semantic similarities with numeric questionnaire signals.

CORRECTIONS v2 :
- Poids corrigés pour correspondre au cahier des charges :
    symptomes=0.60, indications=0.30, numeric=0.10
  (ancienne version : 0.50 / 0.40 / 0.10 — ne correspondait pas au README)
- Normalisation cosine dans [-1, 1] → remappage [0, 1] pour éviter les scores négatifs
- Score numérique enrichi : red_flags cumulatif (0.15 par flag, cap 0.30)
- Bonus de localisation : +0.05 pour les spécialités correspondant à la zone anatomique
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


DUREE_MAP = {
    "moins de 24h": 0.2,
    "1-3 jours":    0.4,
    "1 semaine":    0.6,
    "chronique":    0.8,
}

# Localisation utilisateur → spécialités booostées
LOCALISATION_BOOST = {
    "poitrine":  {"Cardiologie", "Pneumologie"},
    "thorax":    {"Cardiologie", "Pneumologie"},
    "abdomen":   {"Gastroenterologie"},
    "ventre":    {"Gastroenterologie"},
    "tête":      {"Neurologie", "ORL"},
    "tete":      {"Neurologie", "ORL"},
    "gorge":     {"ORL"},
    "dos":       {"Rhumatologie", "Orthopedie"},
    "lombaire":  {"Rhumatologie", "Orthopedie"},
    "genou":     {"Rhumatologie", "Orthopedie"},
    "jambe":     {"Rhumatologie", "Orthopedie"},
    "urines":    {"Urologie", "Nephrologie"},
    "peau":      {"Dermatologie"},
    "oeil":      {"Ophtalmologie"},
    "yeux":      {"Ophtalmologie"},
    "pelvis":    {"Gynecologie"},
    "pelvien":   {"Gynecologie"},
}


def compute_numeric_score(answers: Dict) -> float:
    """
    Numeric score (0-1) based on intensity, duration, and red flags count.
    """
    intensity = answers.get("intensite")
    duree     = (answers.get("duree") or "").strip().lower()
    red_flags = answers.get("red_flags") or []

    score = 0.0
    if intensity is not None:
        score += 0.40 * (max(0, min(int(intensity), 5)) / 5)
    if duree in DUREE_MAP:
        score += 0.30 * DUREE_MAP[duree]
    if red_flags:
        # Cumulatif : 0.15 par flag, plafonné à 0.30
        score += min(0.15 * len(red_flags), 0.30)
    return min(score, 1.0)


def _remap_cosine(sim: np.ndarray) -> np.ndarray:
    """Remap cosine [-1,1] → [0,1] pour éviter les scores globaux négatifs."""
    return (sim + 1.0) / 2.0


def attach_scores(
    df: pd.DataFrame,
    sim_symptomes: np.ndarray,
    sim_indications: np.ndarray,
    numeric_score: float,
    weight_symptomes:   float = 0.60,   # CORRIGÉ (était 0.50)
    weight_indications: float = 0.30,   # CORRIGÉ (était 0.40)
    weight_numeric:     float = 0.10,
    localisation:       str   = "",
) -> pd.DataFrame:
    """
    Add score columns to a referential dataframe and compute global score.
    """
    if not (len(df) == len(sim_symptomes) == len(sim_indications)):
        raise ValueError("Similarity vectors must align with dataframe rows.")

    df_scored = df.copy()

    # Remappage cosine → [0,1]
    df_scored["ScoreSymptomes"]   = _remap_cosine(sim_symptomes)
    df_scored["ScoreIndications"] = _remap_cosine(sim_indications)
    df_scored["ScoreNumerique"]   = numeric_score

    df_scored["ScoreGlobal"] = (
        weight_symptomes   * df_scored["ScoreSymptomes"]
        + weight_indications * df_scored["ScoreIndications"]
        + weight_numeric     * df_scored["ScoreNumerique"]
    )

    # Bonus localisation anatomique
    if localisation:
        loc_lower = localisation.strip().lower()
        boosted = set()
        for key, specs in LOCALISATION_BOOST.items():
            if key in loc_lower:
                boosted |= specs
        if boosted:
            mask = df_scored["Specialite"].isin(boosted)
            df_scored.loc[mask, "ScoreGlobal"] += 0.05

    df_scored = df_scored.sort_values(by="ScoreGlobal", ascending=False).reset_index(drop=True)
    return df_scored
