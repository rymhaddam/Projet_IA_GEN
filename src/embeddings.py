"""
SBERT embeddings helpers.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load and cache the SBERT model. Uses a tiny model to keep resource usage low.
    """
    return SentenceTransformer(model_name)


def embed_texts(texts: Iterable[str], model_name: str = DEFAULT_MODEL) -> np.ndarray:
    """
    Encode an iterable of texts into embeddings matrix (n, d).
    """
    model = get_model(model_name)
    return model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)


def embed_referential_rows(
    symptomes: List[str],
    indications: List[str],
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Compute embeddings for symptoms and indications lists.
    Returns a dict with 'symptomes' and 'indications' matrices aligned to input order.
    """
    return {
        "symptomes": embed_texts(symptomes, model_name=model_name),
        "indications": embed_texts(indications, model_name=model_name),
    }
