"""
SBERT embeddings helpers.
Embedding model: abhinand/MedEmbed-base-v0.1 (medical-domain optimized).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


# MedEmbed-base-v0.1 : modèle d'embeddings spécialisé domaine médical.
# Overrideable via variable d'environnement EMBEDDING_MODEL.
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "abhinand/MedEmbed-base-v0.1")


@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer model.
    Default: abhinand/MedEmbed-base-v0.1 (medical-domain optimized embeddings).
    Overrideable via env var EMBEDDING_MODEL.
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
