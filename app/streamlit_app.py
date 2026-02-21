import os
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline_with_generation


st.set_page_config(page_title="Orientation médicale", page_icon="🩺", layout="wide")
st.title("Assistant d'orientation médicale (RAG + GenAI)")
st.caption("Ceci n'est pas un diagnostic. En cas d'urgence, contactez les services d'urgence immédiatement.")

RED_FLAG_OPTIONS = [
    "douleur thoracique",
    "douleur thoracique intense",
    "essoufflement/dyspnée",
    "gêne respiratoire",
    "hemoptysie",
    "syncope / perte de connaissance",
    "palpitations + malaise",
    "fièvre >39 persistante",
    "altération de la conscience",
    "déficit moteur/sensoriel",
    "trouble de la parole",
    "douleur lombaire avec déficit",
    "douleur abdominale + choc",
    "sang dans les selles",
    "vomissements incoercibles",
    "surdité brutale",
    "baisse visuelle brutale",
    "céphalée en coup de tonnerre",
    "douleur articulaire + fièvre",
    "œdème de Quincke / choc anaphylactique",
    "grossesse + douleurs abdominales intenses",
]

with st.form("input_form"):
    description = st.text_area("Décrivez vos symptômes", height=120)
    intensite = st.slider("Intensité (1-5)", 1, 5, 3)
    duree = st.selectbox("Durée", ["moins de 24h", "1-3 jours", "1 semaine", "chronique"])
    localisation = st.text_input("Localisation (ex: poitrine, tête, bas-ventre)", "general")
    red_flags = st.multiselect("Signaux d'alerte connus", RED_FLAG_OPTIONS)
    red_flag_other = st.text_input("Autre signal d'alerte (optionnel)")
    retrieve_k = st.slider("Passages RAG (k)", 1, 10, 5)
    submitted = st.form_submit_button("Analyser")

if submitted:
    answers = {
        "description": description,
        "intensite": intensite,
        "duree": duree,
        "localisation": localisation,
        "red_flags": red_flags + ([red_flag_other] if red_flag_other else []),
    }
    try:
        out = run_pipeline_with_generation(
            answers,
            retrieve_k=retrieve_k,
        )
    except Exception as e:
        st.error(f"Erreur: {e}")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top spécialités")
            st.dataframe(out["top3"][["Specialite", "ScoreGlobal"]])
            st.subheader("Passages RAG")
            for p in out["retrieved"]:
                st.markdown(f"- **{p.specialite}** (sim {p.score:.2f}) — {p.text}")
        with col2:
            st.subheader("Texte généré")
            st.write(out["genai_text"])
            st.subheader("Plan de progression")
            st.write(out["plan_text"])
            st.subheader("Bio synthétique")
            st.write(out["bio_text"])

st.sidebar.info("Clé API requise (env GOOGLE_API_KEY). Modèle: GENAI_MODEL (par défaut gemini-2.5-flash).")
