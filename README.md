# Projet IA Générative – Orientation médicale (Gemini 2.5)

## Lancer en local
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configurer les variables (copie depuis `.env.example`) :
```bash
export GOOGLE_API_KEY="votre_cle"
export GENAI_PROVIDER=gemini
export GENAI_MODEL=gemini-2.5-flash
```

Exécuter la démo CLI :
```bash
.venv/bin/python -m src.cli_demo \
  --description "douleur thoracique et essoufflement" \
  --intensite 4 \
  --duree "1-3 jours" \
  --localisation poitrine \
  --with-genai
```

Tests :
```bash
.venv/bin/python -m pytest -q
```

## Notes
- Le provider GenAI est limité à Gemini (modèle par défaut `gemini-2.5-flash`).
- Le cache des réponses génératives est stocké dans `data/cache_genai.json`.
