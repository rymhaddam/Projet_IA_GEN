"""
Cosine similarity helpers.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim_matrix(user_vector: np.ndarray, ref_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single vector (d,) and a matrix (n, d).
    Returns shape (n,).
    """
    if user_vector.ndim == 1:
        user_vector = user_vector.reshape(1, -1)
    sims = cosine_similarity(user_vector, ref_matrix)
    return sims.ravel()


def aggregate_similarity(
    user_vector: np.ndarray,
    ref_embeddings: dict,
) -> dict:
    """
    Compute similarities for each embedding matrix in ref_embeddings.
    Returns a dict with same keys and (n,) arrays.
    """
    scores = {}
    for key, matrix in ref_embeddings.items():
        scores[key] = cosine_sim_matrix(user_vector, matrix)
    return scores
