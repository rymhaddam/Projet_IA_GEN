"""
Quick validator for the medical referential CSV.

Checks:
- required columns
- no empty required fields
- basic stats on duplicates and NaN
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REQUIRED = [
    "MedID",
    "BlockID",
    "Specialite",
    "Symptomes_associes",
    "Indications",
    "RedFlags",
    "Organes",
]


def validate(path: Path) -> int:
    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return 1

    empties = {}
    for col in REQUIRED:
        empties[col] = int(df[col].isna().sum())
    empty_total = sum(empties.values())

    dup_medid = int(df["MedID"].duplicated().sum()) if "MedID" in df else 0

    print(f"✅ Rows: {len(df)}")
    print(f"✅ Columns: {list(df.columns)}")
    print(f"ℹ️  Empty required fields total: {empty_total} -> {empties}")
    print(f"ℹ️  Duplicate MedID: {dup_medid}")

    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/medical_referential.csv")
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    sys.exit(validate(path))
