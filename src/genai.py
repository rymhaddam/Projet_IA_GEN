"""
GenAI integration restricted to Gemini (model Gemini 2.5).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
from google.api_core import exceptions as gexc

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore


CACHE_PATH = Path("data/cache_genai.json")
# Default strictly to Gemini 2.5; overrideable via env GENAI_MODEL if necessary.
DEFAULT_MODEL = os.getenv("GENAI_MODEL", "gemini-2.5-flash")


def _json_safe(obj: Any):
    """
    Make common ML/NLP objects JSON-serializable for caching.
    """
    # Custom dataclass for retrieval results
    if hasattr(obj, "__dict__") and all(k in obj.__dict__ for k in ("med_id", "specialite", "text", "score")):
        return {
            "med_id": obj.med_id,
            "specialite": obj.specialite,
            "text": obj.text,
            "score": obj.score,
        }
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _hash_context(context: Dict[str, Any]) -> str:
    payload = json.dumps(context, default=_json_safe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path = CACHE_PATH) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: Dict[str, Any], cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def generate_with_cache(
    context: Dict[str, Any],
    generate_fn: Callable[[Dict[str, Any]], str],
    cache_path: Path = CACHE_PATH,
) -> str:
    """
    Generic cached generation helper.
    """
    cache = load_cache(cache_path)
    key = _hash_context(context)
    if key in cache:
        return cache[key]
    result = generate_fn(context)
    cache[key] = result
    save_cache(cache, cache_path)
    return result


def default_prompt(context: Dict[str, Any]) -> str:
    """
    Build a concise prompt for the medical orientation explanation.
    """
    def _truncate(txt: str, max_len: int = 400) -> str:
        return txt if len(txt) <= max_len else txt[: max_len - 3] + "..."

    top_lines = []
    top3 = context.get("top3", [])
    try:
        iterator = top3.iterrows()
    except AttributeError:
        iterator = enumerate(top3)
    for _, row in iterator:
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        specialite = row.get("Specialite", "N/A")
        score = row.get("ScoreGlobal", 0.0)
        sympt = row.get("Symptomes_associes", "")
        indic = row.get("Indications", "")
        top_lines.append(
            _truncate(f"- {specialite} (score {score:.2f}): symptomes proches={sympt} | indications={indic}", 300)
        )
    # Retrieved passages from RAG
    retrieved_lines = []
    retrieved = context.get("retrieved") or []
    try:
        iterator_ret = retrieved
    except TypeError:
        iterator_ret = []
    for item in iterator_ret:
        try:
            txt = item.text
            sc = item.score
            sp = item.specialite
            mid = item.med_id
        except Exception:
            continue
        retrieved_lines.append(_truncate(f"- [{mid}] {sp} (sim {sc:.2f}) | {txt}", 400))

    redflag = "⚠️ Red flags détectés, rappeler de consulter en urgence." if context.get("red_flags_detected") else ""
    return (
        "Tu es un assistant d’orientation médicale (pas un diagnostic).\n"
        "Explique brièvement pourquoi ces spécialités sont proposées et donne des conseils généraux.\n"
        f"Texte utilisateur: {context.get('user_text','')}\n"
        "Top spécialités:\n" + "\n".join(top_lines) + "\n"
        "Passages pertinents du référentiel:\n" + "\n".join(retrieved_lines) + "\n"
        + redflag
    )


def plan_prompt(context: Dict[str, Any]) -> str:
    return (
        "Tu es un assistant médical. Propose un plan d'action en 3-5 étapes simples pour le patient, "
        "en te basant uniquement sur les passages du référentiel et les spécialités proposées.\n"
        f"Texte utilisateur: {context.get('user_text','')}\n"
        "Passages pertinents:\n" + "\n".join(
            f"- {getattr(p, 'text', '')}" for p in (context.get("retrieved") or [])
        )
    )


def bio_prompt(context: Dict[str, Any]) -> str:
    return (
        "Rédige une courte bio/fiche patient (3-4 phrases) qui résume la situation, les symptômes clés "
        "et la spécialité la plus probable, sans diagnostic. Reste factuel.\n"
        f"Texte utilisateur: {context.get('user_text','')}\n"
        "Top spécialités: " + ", ".join(
            str(row["Specialite"]) for _, row in context.get("top3", []).iterrows()
        )
    )


def gemini_generate(context: Dict[str, Any], model: Optional[str] = None) -> str:
    """
    Provider using Gemini. Requires GOOGLE_API_KEY and google-generativeai installed.
    """
    model = model or os.getenv("GENAI_MODEL", DEFAULT_MODEL)
    if genai is None:
        raise RuntimeError("google-generativeai not installed. Add it to requirements if you want this provider.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set.")
    genai.configure(api_key=api_key)
    prompt = context.get("prompt") or default_prompt(context)
    mdl = genai.GenerativeModel(model)
    resp = mdl.generate_content(prompt)
    return resp.text


def generate_with_provider(context: Dict[str, Any]) -> str:
    """
    Route generation (Gemini only).
    """
    model_override = os.getenv("GENAI_MODEL", DEFAULT_MODEL)
    return gemini_generate(context, model=model_override)


def generate_explanation(context: Dict[str, Any], cache_path: Path | None = None) -> str:
    ctx = dict(context)
    ctx["_kind"] = "explanation"
    return generate_with_cache(ctx, lambda c: generate_with_provider({**c, "prompt": default_prompt(c)}), cache_path=cache_path or CACHE_PATH)


def generate_plan(context: Dict[str, Any], cache_path: Path | None = None) -> str:
    ctx = dict(context)
    ctx["_kind"] = "plan"
    return generate_with_cache(ctx, lambda c: generate_with_provider({**c, "prompt": plan_prompt(c)}), cache_path=cache_path or CACHE_PATH)


def generate_bio(context: Dict[str, Any], cache_path: Path | None = None) -> str:
    ctx = dict(context)
    ctx["_kind"] = "bio"
    return generate_with_cache(ctx, lambda c: generate_with_provider({**c, "prompt": bio_prompt(c)}), cache_path=cache_path or CACHE_PATH)


def generate_all_parallel(context: Dict[str, Any], cache_path: Path | None = None) -> Dict[str, str]:
    """
    Generate explanation, plan, and bio in parallel (or sequentially for simplicity).
    Returns dict with keys: genai_text, plan_text, bio_text
    """
    genai_text = generate_explanation(context, cache_path)
    plan_text = generate_plan(context, cache_path)
    bio_text = generate_bio(context, cache_path)
    return {
        "genai_text": genai_text,
        "plan_text": plan_text,
        "bio_text": bio_text,
    }
