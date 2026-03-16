# 🩺 MedOrient — Agent d'Orientation Médicale Intelligent

> **Projet IA Générative** — Analyse Sémantique pour l'Orientation Médicale  
> Modèle : `gemini-2.5-flash` · Embeddings : `all-MiniLM-L6-v2` · RAG : FAISS

---

## 📌 Présentation

**MedOrient** est un agent intelligent d'orientation médicale basé sur une architecture **RAG (Retrieval-Augmented Generation)**. À partir des symptômes décrits par un utilisateur, il :

1. **Encode** la description en vecteur sémantique via SBERT
2. **Recherche** les entrées les plus proches dans un référentiel médical (FAISS)
3. **Score** les spécialités candidates (similarité cosinus + score numérique)
4. **Génère** une explication, un plan d'action et une bio synthétique via Gemini 2.5

> ⚠️ **Ce système n'est pas un outil de diagnostic médical.** Il oriente vers la spécialité la plus adaptée. En cas d'urgence, appelez le **15 (SAMU)** ou le **112**.

---

## 🏗️ Architecture

```
Saisie utilisateur
       │
       ▼
┌─────────────────┐
│  Preprocessing  │  normalisation, build_user_text
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌───────────────┐
│  SBERT │ │  FAISS (RAG)  │
│Embed.  │ │  top-k pass.  │
└───┬────┘ └───────┬───────┘
    │               │
    └──────┬────────┘
           ▼
  ┌─────────────────┐
  │  Scoring global │  cosine sim (0.6) + numérique (0.1) + RAG (0.3)
  └────────┬────────┘
           ▼
  ┌─────────────────┐
  │  Gemini 2.5     │  explication · plan · bio patient
  └─────────────────┘
```

### Pondération du score global

| Composante | Poids | Description |
|---|---|---|
| Similarité symptômes | 0.60 | Cosine SBERT sur `Symptomes_associes` |
| Similarité indications | 0.30 | Cosine SBERT sur `Indications` |
| Score numérique | 0.10 | Intensité + durée + red flags |

---

## 📁 Structure du projet

```
.
├── app/
│   └── streamlit_app.py       # Interface utilisateur Streamlit
├── data/
│   ├── medical_referential.csv # Référentiel médical (60 entrées, 7 colonnes)
│   ├── faiss_cache/           # Index FAISS mis en cache
│   └── cache_genai.json       # Cache des réponses Gemini
├── scripts/
│   ├── validate_referential.py # Validation du CSV
│   └── benchmark_retrieval.py  # Benchmark du retrieval
├── src/
│   ├── pipeline.py            # Orchestration du pipeline complet
│   ├── data_loader.py         # Chargement et validation du CSV
│   ├── embeddings.py          # Encodage SBERT
│   ├── retrieval.py           # RAG avec FAISS
│   ├── scoring.py             # Calcul des scores
│   ├── similarity.py          # Similarité cosinus
│   ├── recommendation.py      # Top-N + détection red flags
│   ├── preprocessing.py       # Normalisation texte
│   ├── genai.py               # Intégration Gemini + cache
│   └── cli_demo.py            # Interface CLI
├── tests/
│   ├── test_pipeline.py
│   └── test_scoring.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Lancement

### 1. Environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Variables d'environnement

Copiez `.env.example` et renseignez votre clé Gemini :

```bash
cp .env.example .env
```

```bash
# .env
GOOGLE_API_KEY=your_gemini_key_here
GENAI_PROVIDER=gemini
GENAI_MODEL=gemini-2.5-flash
```

> Obtenez une clé gratuite sur [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Interface Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'interface est disponible sur `http://localhost:8501`

### 4. Interface CLI

```bash
python -m src.cli_demo \
  --description "douleur thoracique et essoufflement" \
  --intensite 4 \
  --duree "1-3 jours" \
  --localisation poitrine \
  --with-genai
```

**Options CLI disponibles :**

| Option | Type | Défaut | Description |
|---|---|---|---|
| `--description` | str | requis | Texte libre des symptômes |
| `--intensite` | int (1-5) | 3 | Niveau de douleur |
| `--duree` | str | `1-3 jours` | Durée des symptômes |
| `--localisation` | str | `general` | Partie du corps |
| `--red-flag` | str (repeatable) | `[]` | Signaux d'alerte (répétable) |
| `--with-genai` | flag | false | Active la génération Gemini |
| `--retrieve-k` | int | 5 | Nombre de passages RAG |
| `--retrieve-rebuild` | flag | false | Reconstruit l'index FAISS |

---

## 🧪 Tests

```bash
# Lancer tous les tests
python -m pytest -q

# Avec couverture
python -m pytest --tb=short -v
```

Les tests couvrent :
- `test_pipeline.py` — pipeline complet avec mock GenAI + vérification du cache
- `test_scoring.py` — bornes et monotonie du score numérique

---

## 🗂️ Référentiel médical

Le fichier `data/medical_referential.csv` contient **60 entrées** couvrant 18 spécialités :

| Spécialité | Entrées |
|---|---|
| Cardiologie | 7 |
| Pneumologie | 5 |
| Neurologie | 6 |
| Gastroentérologie | 6 |
| Infectiologie | 4 |
| Endocrinologie | 4 |
| Rhumatologie | 2 |
| Dermatologie | 3 |
| Pédiatrie | 3 |
| Psychiatrie | 2 |
| Autres (ORL, Ophtalmo, Ortho…) | 18 |

**Colonnes du référentiel :**

| Colonne | Description |
|---|---|
| `MedID` | Identifiant unique (M01…M60) |
| `BlockID` | Groupe thématique |
| `Specialite` | Spécialité médicale cible |
| `Symptomes_associes` | Description textuelle des symptômes (encodé par SBERT) |
| `Indications` | Contexte clinique d'orientation |
| `RedFlags` | Signaux d'urgence à surveiller |
| `Organes` | Organes / systèmes concernés |

**Valider le référentiel :**

```bash
python scripts/validate_referential.py data/medical_referential.csv
```

**Reconstruire l'index FAISS après modification du CSV :**

```bash
python -m src.cli_demo --description "test" --retrieve-rebuild
# ou supprimer manuellement : rm data/faiss_cache/*
```

---

## 🤖 Sorties GenAI

Trois types de textes sont générés par Gemini 2.5 Flash pour chaque analyse :

| Sortie | Description |
|---|---|
| **Explication** | Justification des spécialités proposées + conseils généraux |
| **Plan d'action** | 3 à 5 étapes concrètes pour le patient |
| **Bio synthétique** | Résumé factuel de la situation clinique (3-4 phrases) |

Les réponses sont **mises en cache** dans `data/cache_genai.json` (hash SHA-256 du contexte). Un même contexte ne consomme l'API qu'une seule fois.

---

## ⚠️ Détection des red flags

L'agent détecte automatiquement les signaux d'urgence dans :
- Les cases cochées par l'utilisateur (multiselect)
- Le texte libre de description (recherche lexicale)

Signaux détectés : `douleur thoracique`, `perte de connaissance`, `gêne respiratoire`, `hémoptysie`…

En cas de détection, un **avertissement d'urgence** est affiché et inclus dans le prompt Gemini.

---

## 📦 Dépendances principales

```
sentence-transformers  # SBERT embeddings
faiss-cpu              # Index vectoriel
google-generativeai    # API Gemini
streamlit              # Interface web
pandas / numpy         # Manipulation données
scikit-learn           # Similarité cosinus
pytest                 # Tests
```

---

## 🔒 Notes importantes

- Ce projet utilise **uniquement Gemini** comme provider GenAI (pas d'OpenAI, pas d'Anthropic).
- Le modèle par défaut est `gemini-2.5-flash`, surpassable via `GENAI_MODEL`.
- Aucune donnée patient n'est persistée — le cache stocke uniquement les réponses GenAI.
- Le système est conçu pour un usage académique et de démonstration.

---

## 👥 Équipe

Projet réalisé dans le cadre du cours **IA Générative** — Thème médical choisi.

---

*MedOrient — Agent d'Orientation Médicale · Gemini 2.5 Flash · FAISS · SBERT*
