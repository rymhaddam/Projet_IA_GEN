"""
Lightweight retrieval layer using FAISS with SBERT embeddings.
Includes on-disk caching keyed by referential hash + model.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Iterable

import faiss
import numpy as np
import pandas as pd

from src.embeddings import embed_texts

CACHE_DIR = Path("data/faiss_cache")


@dataclass
class RetrievedPassage:
    med_id: str
    specialite: str
    text: str
    score: float


def _build_passage_text(row: pd.Series) -> str:
    parts = [
        f"Specialite: {row['Specialite']}",
        f"Symptomes: {row['Symptomes_associes']}",
        f"Indications: {row['Indications']}",
        f"RedFlags: {row['RedFlags']}",
        f"Organes: {row['Organes']}",
    ]
    return " | ".join(parts)


def _file_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.md5(data).hexdigest()


def _safe_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in model_name)


def _cache_paths(ref_path: Path, model_name: str, ref_hash: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{ref_path.stem}_{_safe_name(model_name)}_{ref_hash}"
    index_path = CACHE_DIR / f"{base}.index"
    meta_path = CACHE_DIR / f"{base}.json"
    return index_path, meta_path


def _load_cache(index_path: Path, meta_path: Path):
    if not index_path.exists() or not meta_path.exists():
        return None
    try:
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text())
        return index, meta["med_ids"], meta["specials"], meta["texts"]
    except Exception:
        return None


def _save_cache(index, med_ids, specials, texts, index_path: Path, meta_path: Path):
    faiss.write_index(index, str(index_path))
    meta = {"med_ids": med_ids, "specials": specials, "texts": texts}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False))


def build_index(
    df: pd.DataFrame,
    model_name: str,
    ref_path: str | Path,
    force_rebuild: bool = False,
) -> tuple[faiss.IndexFlatIP, List[str], List[str], List[str]]:
    ref_path = Path(ref_path)
    ref_hash = _file_hash(ref_path)
    index_path, meta_path = _cache_paths(ref_path, model_name, ref_hash)

    cached = None if force_rebuild else _load_cache(index_path, meta_path)
    if cached:
        return cached

    texts = [_build_passage_text(row) for _, row in df.iterrows()]
    embeddings = embed_texts(texts, model_name=model_name).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    med_ids = df["MedID"].tolist()
    specials = df["Specialite"].tolist()
    _save_cache(index, med_ids, specials, texts, index_path, meta_path)
    return index, med_ids, specials, texts


def retrieve(
    query: str,
    df: pd.DataFrame,
    model_name: str,
    k: int = 5,
    ref_path: str | Path = "data/medical_referential.csv",
    force_rebuild: bool = False,
    max_passage_chars: Optional[int] = 400,
    filter_fn: Optional[Callable[[pd.Series], bool]] = None,
) -> List[RetrievedPassage]:
    """
    Embed the query, search FAISS, return top-k passages with metadata.
    - force_rebuild: ignore cache and rebuild index
    - max_passage_chars: truncate passages to control prompt size
    - filter_fn: optional row filter before indexing (row: pd.Series -> bool)
    """
    if filter_fn:
        df = df[df.apply(filter_fn, axis=1)]

    index, med_ids, specials, texts = build_index(
        df, model_name=model_name, ref_path=ref_path, force_rebuild=force_rebuild
    )
    q_emb = embed_texts([query], model_name=model_name).astype(np.float32)
    scores, idxs = index.search(q_emb, k)
    out: List[RetrievedPassage] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        txt = texts[idx]
        if max_passage_chars and len(txt) > max_passage_chars:
            txt = txt[: max_passage_chars - 3] + "..."
        out.append(
            RetrievedPassage(
                med_id=med_ids[idx],
                specialite=specials[idx],
                text=txt,
                score=float(score),
            )
        )
    return out
