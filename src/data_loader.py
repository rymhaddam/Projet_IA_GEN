"""
Utilities to load and validate the medical referential CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "MedID",
    "BlockID",
    "Specialite",
    "Symptomes_associes",
    "Indications",
    "RedFlags",
    "Organes",
]


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> List[str]:
    existing = set(df.columns)
    return [c for c in required if c not in existing]


def load_medical_referential(path: str | Path) -> pd.DataFrame:
    """
    Load the referential CSV and ensure required columns are present.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Referential not found at {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='skip')
    missing = _missing_columns(df, REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"Missing required columns in referential: {missing}")
    # Drop fully empty rows to avoid noise
    df = df.dropna(how="all")
    return df
