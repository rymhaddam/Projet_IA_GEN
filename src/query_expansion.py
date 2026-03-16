"""
F4.1 — Augmentation conditionnelle de la saisie utilisateur.

La consigne demande :
  "Vous pouvez développer une fonction qui utilise le LLM pour enrichir la phrase
   [utilisateur] lorsque celle-ci est trop courte."
  Exigence : appel API conditionnel — uniquement si le texte est trop court.

Stratégie :
  - Seuil : < MIN_CHARS caractères OU < MIN_WORDS mots significatifs
  - Si le texte est trop court → un appel Gemini enrichit la description
  - Si le texte est suffisant → aucun appel, la description est utilisée telle quelle
  - L'enrichissement est mis en cache (même clé de cache que le reste)
  - Le texte original est toujours conservé ; l'enrichi est additionnel

Usage dans pipeline.py :
    from src.query_expansion import maybe_expand_query
    expanded = maybe_expand_query(answers)
    answers_enriched = {**answers, "description": expanded}
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# ── Seuils de détection "texte trop court" ────────────────────────────────────
MIN_CHARS = 30    # moins de 30 caractères → trop court
MIN_WORDS = 5     # moins de 5 mots → trop court

# ── Mots fonctionnels à exclure du comptage ───────────────────────────────────
_STOP_WORDS = {
    "j", "ai", "j'ai", "le", "la", "les", "un", "une", "des", "et", "ou",
    "de", "du", "en", "au", "aux", "je", "il", "elle", "on", "nous", "vous",
    "ils", "me", "se", "sa", "mon", "ma", "avec", "sans", "sur", "sous",
    "dans", "par", "pour", "qui", "que", "quoi", "très", "plus", "peu",
}


def _is_too_short(text: str) -> bool:
    """
    Return True if the text is considered too short for reliable embedding.

    A text is too short if EITHER:
      - It has fewer than MIN_CHARS characters, OR
      - It has fewer than MIN_WORDS non-stopword tokens
    """
    if not text or len(text.strip()) < MIN_CHARS:
        return True

    tokens = [w.lower().strip(".,;:!?") for w in text.split()]
    meaningful = [t for t in tokens if t and t not in _STOP_WORDS]
    return len(meaningful) < MIN_WORDS


def _build_expansion_prompt(description: str, answers: Dict) -> str:
    """
    Build a prompt that asks Gemini to expand/enrich a short symptom description
    while remaining medically neutral and factual.
    """
    localisation = answers.get("localisation") or ""
    duree = answers.get("duree") or ""
    intensite = answers.get("intensite")
    age = answers.get("age_tranche") or ""

    context_parts = []
    if localisation:
        context_parts.append(f"localisation : {localisation}")
    if duree:
        context_parts.append(f"durée : {duree}")
    if intensite is not None:
        context_parts.append(f"intensité : {intensite}/5")
    if age:
        context_parts.append(f"profil : {age}")

    context_str = " | ".join(context_parts) if context_parts else "non précisé"

    return (
        "Tu es un assistant médical. Un patient a décrit ses symptômes de façon très brève.\n"
        "Enrichis cette description en une ou deux phrases cliniques claires, "
        "en ajoutant des détails médicaux plausibles et cohérents basés sur le contexte fourni.\n"
        "NE pose PAS de diagnostic. Reste factuel et neutre.\n"
        "Retourne UNIQUEMENT la description enrichie, sans explication supplémentaire.\n\n"
        f"Description originale : \"{description}\"\n"
        f"Contexte : {context_str}"
    )


def expand_query(description: str, answers: Dict) -> str:
    """
    Call Gemini to enrich a short symptom description.
    Uses the existing genai cache infrastructure to avoid redundant API calls.

    Returns the enriched description string.
    Raises RuntimeError if Gemini is unavailable.
    """
    from src.genai import (
        CACHE_PATH,
        generate_with_cache,
        generate_with_provider,
    )

    prompt = _build_expansion_prompt(description, answers)

    # Cache key includes the original description + context to avoid collisions
    cache_context = {
        "_kind": "query_expansion",
        "description": description,
        "localisation": answers.get("localisation") or "",
        "duree": answers.get("duree") or "",
        "intensite": answers.get("intensite"),
        "age_tranche": answers.get("age_tranche") or "",
    }

    expanded = generate_with_cache(
        cache_context,
        lambda _ctx: generate_with_provider({**_ctx, "prompt": prompt}),
        cache_path=CACHE_PATH,
    )

    logger.info(
        "Query expanded: [%d chars] → [%d chars]",
        len(description),
        len(expanded),
    )
    return expanded.strip()


def maybe_expand_query(answers: Dict) -> str:
    """
    F4.1 — Conditional query expansion.

    If the user's symptom description is too short, call Gemini to enrich it.
    Otherwise return the original description unchanged.

    Args:
        answers: questionnaire answers dict (must contain 'description').

    Returns:
        The (possibly enriched) symptom description string.
    """
    description: str = (answers.get("description") or "").strip()

    if not _is_too_short(description):
        # Text is long enough — no API call needed
        return description

    logger.info(
        "Description too short (%d chars, %d words) — triggering F4.1 expansion.",
        len(description),
        len(description.split()),
    )

    try:
        expanded = expand_query(description, answers)
        # Safety check: if expansion is empty or shorter, keep original
        if expanded and len(expanded) > len(description):
            return expanded
        logger.warning("Expansion result was not longer than original — keeping original.")
        return description
    except Exception as exc:
        # Never block the pipeline if expansion fails
        logger.warning("Query expansion failed (%s) — using original description.", exc)
        return description
