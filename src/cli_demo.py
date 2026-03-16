"""
Minimal CLI demo to run the pipeline without Streamlit.
"""
from __future__ import annotations

import argparse
import json

from src.pipeline import run_pipeline, run_pipeline_with_generation


def parse_args():
    p = argparse.ArgumentParser(description="Run medical orientation pipeline (CLI demo).")
    p.add_argument("--description", required=True, help="Texte libre des symptômes")
    p.add_argument("--intensite", type=int, default=3)
    p.add_argument("--duree", default="1-3 jours")
    p.add_argument("--localisation", default="general")
    p.add_argument("--red-flag", action="append", dest="red_flags", default=[])
    p.add_argument("--ref", default="data/medical_referential.csv")
    p.add_argument("--with-genai", action="store_true", help="Activer la génération GenAI (clé requise si provider cloud).")
    p.add_argument("--retrieve-k", type=int, default=5, help="Nombre de passages RAG à récupérer.")
    p.add_argument("--retrieve-rebuild", action="store_true", help="Reconstruire l'index FAISS en ignorant le cache.")
    return p.parse_args()


def main():
    args = parse_args()
    answers = {
        "description": args.description,
        "intensite": args.intensite,
        "duree": args.duree,
        "localisation": args.localisation,
        "red_flags": args.red_flags,
    }
    common_kwargs = {
        "ref_path": args.ref,
        "retrieve_k": args.retrieve_k,
        "retrieve_force_rebuild": args.retrieve_rebuild,
    }
    if args.with_genai:
        out = run_pipeline_with_generation(answers, **common_kwargs)
    else:
        out = run_pipeline(answers, **common_kwargs)
    print("Texte utilisateur:", out["user_text"])
    print("\nTop 3 spécialités:")
    print(out["top3"][["Specialite", "ScoreGlobal"]])
    if out["red_flags_detected"]:
        print("\n⚠️ Red flags détectés:", out["red_flags_hits"])
    print("\nDétails (JSON):")
    print(json.dumps({
        "numeric_score": out["numeric_score"],
        "top3": out["top3"][["Specialite", "ScoreGlobal"]].to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))
    if args.with_genai:
        print("\nTexte généré (GenAI):")
        print(out["genai_text"])
        print("\nPlan de progression (GenAI):")
        print(out["plan_text"])
        print("\nBio synthétique (GenAI):")
        print(out["bio_text"])


if __name__ == "__main__":
    main()
