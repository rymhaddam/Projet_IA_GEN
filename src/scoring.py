"""
Score computation: combine semantic similarities with numeric questionnaire signals.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


DUREE_MAP = {
    "moins de 24h": 0.2,
    "1-3 jours": 0.4,
    "1 semaine": 0.6,
    "chronique": 0.8,
}


def compute_numeric_score(answers: Dict) -> float:
    """
    Simple numeric score (0-1) based on intensity, duration, and red flags.
    """
    intensity = answers.get("intensite")
    duree = (answers.get("duree") or "").strip().lower()
    red_flags = answers.get("red_flags") or []

    score = 0.0
    if intensity is not None:
        score += 0.4 * (max(0, min(intensity, 5)) / 5)
    if duree in DUREE_MAP:
        score += 0.3 * DUREE_MAP[duree]
    if red_flags:
        score += 0.3
    return min(score, 1.0)


def attach_scores(
    df: pd.DataFrame,
    sim_symptomes: np.ndarray,
    sim_indications: np.ndarray,
    numeric_score: float,
    weight_symptomes: float = 0.5,
    weight_indications: float = 0.4,
    weight_numeric: float = 0.1,
) -> pd.DataFrame:
    """
    Add score columns to a referential dataframe and compute global score.
    """
    if not (len(df) == len(sim_symptomes) == len(sim_indications)):
        raise ValueError("Similarity vectors must align with dataframe rows.")

    df_scored = df.copy()
    df_scored["ScoreSymptomes"] = sim_symptomes
    df_scored["ScoreIndications"] = sim_indications
    df_scored["ScoreNumerique"] = numeric_score
    df_scored["ScoreGlobal"] = (
        weight_symptomes * df_scored["ScoreSymptomes"]
        + weight_indications * df_scored["ScoreIndications"]
        + weight_numeric * df_scored["ScoreNumerique"]
    )
    df_scored = df_scored.sort_values(by="ScoreGlobal", ascending=False).reset_index(drop=True)
    return df_scored
