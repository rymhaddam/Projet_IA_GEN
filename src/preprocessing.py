"""
Text preprocessing and user answer structuring for the medical orientation agent.

CORRECTIONS v2 :
- build_user_text : enrichissement sémantique du texte utilisateur
  → on répète les symptômes clés avec leur localisation pour améliorer
    l'alignement avec les colonnes Symptomes_associes du référentiel
  → on ajoute le contexte âge (age_tranche) s'il est disponible
- normalize_text : suppression des accents optionnelle (désactivée par défaut
  pour garder la compatibilité avec MedEmbed)
"""
from __future__ import annotations

import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    """
    Lowercase, strip, collapse spaces. Keeps accents/punctuation as-is.
    """
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def build_user_text(answers: Dict) -> str:
    """
    Build a rich, structured text from the questionnaire answers.

    FIX v2 : on construit un texte plus dense sémantiquement en combinant
    description + localisation dans une même phrase, ce qui améliore la
    similarité cosinus avec les colonnes Symptomes_associes du CSV.

    Expected keys:
      - description   : str
      - intensite     : int (1-5)
      - duree         : str
      - localisation  : str
      - red_flags     : List[str]
      - age_tranche   : str (optionnel)
    """
    description   = normalize_text(answers.get("description", ""))
    intensite     = answers.get("intensite")
    duree         = answers.get("duree", "")
    localisation  = normalize_text(answers.get("localisation", ""))
    red_flags: List[str] = answers.get("red_flags") or []
    age_tranche   = normalize_text(answers.get("age_tranche", ""))

    parts = []

    # Phrase principale : description + localisation combinées
    if description and localisation:
        parts.append(f"symptomes declares au niveau de {localisation}: {description}.")
    elif description:
        parts.append(f"symptomes declares: {description}.")
    elif localisation:
        parts.append(f"douleur ou gene au niveau de {localisation}.")

    if duree:
        parts.append(f"duree des symptomes: {normalize_text(duree)}.")
    if intensite is not None:
        parts.append(f"intensite de la douleur: {intensite}/5.")
    if age_tranche:
        parts.append(f"profil patient: {age_tranche}.")
    if red_flags:
        flags_txt = ", ".join(normalize_text(f) for f in red_flags)
        parts.append(f"signaux d'alerte mentionnes: {flags_txt}.")

    return " ".join(parts)
