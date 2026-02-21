"""
Core pipeline orchestrating preprocessing, embeddings, similarity, and scoring.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.data_loader import load_medical_referential
from src.embeddings import DEFAULT_MODEL, embed_referential_rows, embed_texts
from src.preprocessing import build_user_text
from src.recommendation import detect_red_flags, top_specialties
from src.scoring import attach_scores, compute_numeric_score
from src.similarity import aggregate_similarity
from src.retrieval import retrieve


@lru_cache(maxsize=1)
def _referential_embeddings(ref_path: str, model_name: str = DEFAULT_MODEL):
    """
    Load referential and compute embeddings once (cached).
    """
    df = load_medical_referential(ref_path)
    embeds = embed_referential_rows(
        symptomes=df["Symptomes_associes"].tolist(),
        indications=df["Indications"].tolist(),
        model_name=model_name,
    )
    return df, embeds


def run_pipeline(
    answers: Dict,
    ref_path: str = "data/medical_referential.csv",
    model_name: str = DEFAULT_MODEL,
    retrieve_k: int = 5,
    retrieve_force_rebuild: bool = False,
) -> Dict:
    """
    Execute the full NLP + scoring pipeline for one user input.
    Returns a dict with scored dataframe, top3, user_text, red_flags, retrieval hits.
    """
    user_text = build_user_text(answers)
    user_embedding = embed_texts([user_text], model_name=model_name)[0]

    df_ref, ref_embeddings = _referential_embeddings(ref_path, model_name)
    sims = aggregate_similarity(user_embedding, ref_embeddings)

    numeric_score = compute_numeric_score(answers)
    df_scored = attach_scores(
        df=df_ref,
        sim_symptomes=sims["symptomes"],
        sim_indications=sims["indications"],
        numeric_score=numeric_score,
    )

    top3 = top_specialties(df_scored, n=3)
    critical, flags = detect_red_flags(answers, answers.get("description", ""))
    retrieved = retrieve(
        user_text,
        df_ref,
        model_name=model_name,
        k=retrieve_k,
        ref_path=ref_path,
        force_rebuild=retrieve_force_rebuild,
    )

    return {
        "user_text": user_text,
        "scores": df_scored,
        "top3": top3,
        "numeric_score": numeric_score,
        "red_flags_detected": critical,
        "red_flags_hits": flags,
        "retrieved": retrieved,
    }


def run_pipeline_with_generation(
    answers: Dict,
    ref_path: str = "data/medical_referential.csv",
    model_name: str = DEFAULT_MODEL,
    cache_path: str | None = None,
    retrieve_k: int = 5,
    retrieve_force_rebuild: bool = False,
) -> Dict:
    """
    Execute pipeline then trigger GenAI generation with caching.

    cache_path allows tests to isolate cache writes.
    """
    from src.genai import CACHE_PATH, generate_with_cache, generate_with_provider
    from src.genai import generate_explanation, generate_plan, generate_bio

    base = run_pipeline(
        answers,
        ref_path=ref_path,
        model_name=model_name,
        retrieve_k=retrieve_k,
        retrieve_force_rebuild=retrieve_force_rebuild,
    )
    context = {
        "user_text": base["user_text"],
        "top3": base["top3"],
        "red_flags_detected": base["red_flags_detected"],
        "retrieved": base["retrieved"],
    }

    selected_cache = CACHE_PATH if cache_path is None else Path(cache_path)
    gen_text = generate_explanation(context, cache_path=selected_cache)
    plan_text = generate_plan(context, cache_path=selected_cache)
    bio_text = generate_bio(context, cache_path=selected_cache)

    base["genai_text"] = gen_text
    base["plan_text"] = plan_text
    base["bio_text"] = bio_text
    return base
