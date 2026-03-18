"""
Core pipeline orchestrating preprocessing, embeddings, similarity, and scoring.

CORRECTIONS v2 :
- attach_scores reçoit maintenant la localisation → bonus anatomique actif
- top_specialties utilise la version corrigée (déduplication par spécialité)
- Poids scoring transmis explicitement pour traçabilité
- run_pipeline retourne aussi 'localisation' dans le résultat
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


class PipelineResult(TypedDict):
    user_text:            str
    description_expanded: bool
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
_REF_CACHE: Dict[tuple[str, str], tuple[pd.DataFrame, dict]] = {}


def _get_referential_embeddings(
    ref_path: str,
    model_name: str = DEFAULT_MODEL,
) -> tuple[pd.DataFrame, dict]:
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
    _REF_CACHE.clear()
    logger.info("Referential embedding cache cleared.")


def run_pipeline(
    answers: Dict,
    ref_path: str = "data/medical_referential.csv",
    model_name: str = DEFAULT_MODEL,
    retrieve_k: int = 5,
    retrieve_force_rebuild: bool = False,
) -> PipelineResult:
    """
    Execute the full NLP + scoring pipeline for one user input.
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

    # 3. Embed user query
    user_embedding: np.ndarray = embed_texts([user_text], model_name=model_name)[0]

    # 4. Referential (cached)
    df_ref, ref_embeddings = _get_referential_embeddings(ref_path, model_name)

    # 5. Similarities
    sims = aggregate_similarity(user_embedding, ref_embeddings)

    # 6. Numeric score
    numeric_score: float = compute_numeric_score(answers)

    # 7. Scoring — FIX : passer la localisation pour le bonus anatomique
    localisation = (answers.get("localisation") or "").strip()
    df_scored: pd.DataFrame = attach_scores(
        df=df_ref,
        sim_symptomes=sims["symptomes"],
        sim_indications=sims["indications"],
        numeric_score=numeric_score,
        localisation=localisation,   # ← NOUVEAU
    )

    # 8. Top-3 unique specialties (FIX : déduplication)
    top3: pd.DataFrame = top_specialties(df_scored, n=3)

    # 9. Red flags
    critical, flags = detect_red_flags(answers, answers.get("description", ""))

    # 10. RAG retrieval
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
