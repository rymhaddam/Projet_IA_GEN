"""
Text preprocessing and user answer structuring for the medical orientation agent.
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
    Build a concise, structured text from the questionnaire answers.

    Expected keys in answers:
      - description: str
      - intensite: int (1-5)
      - duree: str
      - localisation: str
      - red_flags: List[str]
    """
    description = normalize_text(answers.get("description", ""))
    intensite = answers.get("intensite")
    duree = answers.get("duree", "")
    localisation = answers.get("localisation", "")
    red_flags: List[str] = answers.get("red_flags") or []

    parts = []
    if description:
        parts.append(f"symptomes declares: {description}.")
    if localisation:
        parts.append(f"localisation: {normalize_text(localisation)}.")
    if duree:
        parts.append(f"duree: {normalize_text(duree)}.")
    if intensite is not None:
        parts.append(f"intensite: {intensite}/5.")
    if red_flags:
        flags_txt = ", ".join(normalize_text(f) for f in red_flags)
        parts.append(f"signaux d'alerte mentionnes: {flags_txt}.")
    return " ".join(parts)
