"""
Core pipeline orchestrating preprocessing, embeddings, similarity, and scoring.

Optimisations vs version originale :
- _referential_embeddings utilise un vrai cache module-level dict
  (key = (ref_path, model_name)) pour supporter plusieurs référentiels
- run_pipeline_with_generation utilise generate_all_parallel → 3× plus rapide
- Typage complet avec TypedDict pour les résultats
- Invalidation explicite du cache via clear_referential_cache()
- F4.1 : enrichissement conditionnel de la saisie utilisateur via maybe_expand_query
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, TypedDict

import numpy as np
import pandas as pd

from src.data_loader import load_medical_referential
from src.embeddings import DEFAULT_MODEL, embed_referential_rows, embed_texts
from src.preprocessing import build_user_text
from src.query_expansion import maybe_expand_query
from src.recommendation import detect_red_flags, top_specialties
from src.retrieval import RetrievedPassage, retrieve
from src.scoring import attach_scores, compute_numeric_score
from src.similarity import aggregate_similarity

logger = logging.getLogger(__name__)


# ── TypedDicts for clear return contracts ────────────────────────────────────

class PipelineResult(TypedDict):
    user_text:            str
    description_expanded: bool   # True si F4.1 a enrichi la description
    scores:               pd.DataFrame
    top3:                 pd.DataFrame
    numeric_score:        float
    red_flags_detected:   bool
    red_flags_hits:       list[str]
    retrieved:            list[RetrievedPassage]


class GenerationResult(PipelineResult):
    genai_text: str
    plan_text:  str
    bio_text:   str


# ── Referential embedding cache ───────────────────────────────────────────────
# key: (ref_path_str, model_name) → (df, embeddings_dict)
_REF_CACHE: Dict[tuple[str, str], tuple[pd.DataFrame, dict]] = {}


def _get_referential_embeddings(
    ref_path: str,
    model_name: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, dict]:
    """
    Return cached (df, embeddings). Reloads only when ref_path or model changes.
    Supports multiple referentials (unlike lru_cache(maxsize=1)).
    """
    cache_key = (str(ref_path), model_name)
    if cache_key not in _REF_CACHE:
        logger.info("Loading referential and computing embeddings: %s", ref_path)
        df = load_medical_referential(ref_path)
        embeds = embed_referential_rows(
            symptomes=df["Symptomes_associes"].tolist(),
            indications=df["Indications"].tolist(),
            model_name=model_name,
        )
        _REF_CACHE[cache_key] = (df, embeds)
    return _REF_CACHE[cache_key]


def clear_referential_cache() -> None:
    """
    Evict all cached referential embeddings.
    Call after updating the CSV to force re-encoding on next pipeline run.
    """
    _REF_CACHE.clear()
    logger.info("Referential embedding cache cleared.")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    answers: Dict,
    ref_path: str = "data/medical_referential.csv",
    model_name: str = DEFAULT_MODEL,
    retrieve_k: int = 5,
    retrieve_force_rebuild: bool = False,
) -> PipelineResult:
    """
    Execute the full NLP + scoring pipeline for one user input.

    Steps:
        1. Preprocess user input → structured text
        2. Embed user text (SBERT)
        3. Load referential embeddings (cached)
        4. Compute cosine similarities
        5. Compute numeric severity score
        6. Attach & rank scores
        7. Select top-3 unique specialties
        8. Detect red flags
        9. Retrieve top-k RAG passages (FAISS)

    Args:
        answers:               questionnaire answers dict.
        ref_path:              path to the medical referential CSV.
        model_name:            SBERT model identifier.
        retrieve_k:            number of RAG passages to retrieve.
        retrieve_force_rebuild: bypass FAISS cache.

    Returns:
        PipelineResult dict.
    """
    # 1. F4.1 — Enrichissement conditionnel si description trop courte
    original_desc = (answers.get("description") or "").strip()
    expanded_desc = maybe_expand_query(answers)
    description_was_expanded = expanded_desc != original_desc

    if description_was_expanded:
        answers = {**answers, "description": expanded_desc}
        logger.info("F4.1 active: description enrichie (%d → %d chars).",
                    len(original_desc), len(expanded_desc))

    # 2. Preprocess
    user_text: str = build_user_text(answers)
    logger.debug("User text: %s", user_text)

    # 2. Embed user query
    user_embedding: np.ndarray = embed_texts([user_text], model_name=model_name)[0]

    # 3. Referential (cached)
    df_ref, ref_embeddings = _get_referential_embeddings(ref_path, model_name)

    # 4. Similarities
    sims = aggregate_similarity(user_embedding, ref_embeddings)

    # 5. Numeric score
    numeric_score: float = compute_numeric_score(answers)

    # 6. Scoring
    df_scored: pd.DataFrame = attach_scores(
        df=df_ref,
        sim_symptomes=sims["symptomes"],
        sim_indications=sims["indications"],
        numeric_score=numeric_score,
    )

    # 7. Top-3 unique specialties
    top3: pd.DataFrame = top_specialties(df_scored, n=3)

    # 8. Red flags
    critical, flags = detect_red_flags(answers, answers.get("description", ""))

    # 9. RAG retrieval
    retrieved = retrieve(
        query=user_text,
        df=df_ref,
        model_name=model_name,
        k=retrieve_k,
        ref_path=ref_path,
        force_rebuild=retrieve_force_rebuild,
    )

    return PipelineResult(
        user_text=user_text,
        description_expanded=description_was_expanded,
        scores=df_scored,
        top3=top3,
        numeric_score=numeric_score,
        red_flags_detected=critical,
        red_flags_hits=flags,
        retrieved=retrieved,
    )


def run_pipeline_with_generation(
    answers: Dict,
    ref_path: str = "data/medical_referential.csv",
    model_name: str = DEFAULT_MODEL,
    cache_path: Optional[str | Path] = None,
    retrieve_k: int = 5,
    retrieve_force_rebuild: bool = False,
) -> GenerationResult:
    """
    Execute the NLP pipeline then generate all three GenAI outputs in parallel.

    The three Gemini calls (explanation, plan, bio) run concurrently via
    ThreadPoolExecutor, reducing wall-clock latency by ~3× compared to
    sequential calls.

    Args:
        answers:               questionnaire answers dict.
        ref_path:              path to the medical referential CSV.
        model_name:            SBERT model identifier.
        cache_path:            override GenAI cache path (useful in tests).
        retrieve_k:            number of RAG passages.
        retrieve_force_rebuild: bypass FAISS cache.

    Returns:
        GenerationResult dict (PipelineResult + genai_text, plan_text, bio_text).
    """
    from src.genai import CACHE_PATH, generate_all_parallel

    base = run_pipeline(
        answers,
        ref_path=ref_path,
        model_name=model_name,
        retrieve_k=retrieve_k,
        retrieve_force_rebuild=retrieve_force_rebuild,
    )

    context = {
        "user_text":          base["user_text"],
        "top3":               base["top3"],
        "red_flags_detected": base["red_flags_detected"],
        "retrieved":          base["retrieved"],
    }

    resolved_cache = Path(cache_path) if cache_path else CACHE_PATH
    generated = generate_all_parallel(context, cache_path=resolved_cache)

    return GenerationResult(
        **base,
        genai_text=generated.get("genai_text", ""),
        plan_text=generated.get("plan_text", ""),
        bio_text=generated.get("bio_text", ""),
    )
