"""
Simple retrieval benchmark:
- measures retrieval latency over N runs
- prints top passages for a given query
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval import retrieve
from src.embeddings import DEFAULT_MODEL


def main():
    ref_path = Path("data/medical_referential.csv")
    df = pd.read_csv(ref_path)
    query = "fièvre avec frissons, toux sèche, courbatures"
    runs = 5

    # warmup
    retrieve(query, df, model_name=DEFAULT_MODEL, k=5, ref_path=ref_path)

    t0 = time.time()
    for _ in range(runs):
        res = retrieve(query, df, model_name=DEFAULT_MODEL, k=5, ref_path=ref_path)
    dt = (time.time() - t0) / runs

    print(f"Moyenne par requête: {dt*1000:.1f} ms")
    for r in res:
        print(f"- {r.med_id} | {r.specialite} | sim={r.score:.2f} | {r.text[:120]}...")


if __name__ == "__main__":
    main()
