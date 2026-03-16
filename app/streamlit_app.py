import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline_with_generation

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedOrient — Orientation médicale IA",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS global
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700&display=swap');

:root {
    --blue-900:#0A2E52; --blue-800:#0F4C7A; --blue-600:#1B6CA8;
    --blue-400:#3B8BD4; --blue-100:#DBEEFF; --blue-50:#EBF4FF;
    --teal-500:#0D9488; --teal-100:#CCFBF1;
    --red-600:#DC2626;  --red-100:#FEE2E2;
    --amber-500:#D97706;--amber-100:#FEF3C7;
    --green-600:#059669;--green-100:#D1FAE5;
    --gray-950:#0A0A0B; --gray-800:#1C1C1E; --gray-700:#3A3A3C;
    --gray-500:#636366; --gray-300:#C7C7CC; --gray-200:#E5E5EA;
    --gray-100:#F2F2F7; --gray-50:#F9F9FB;  --white:#FFFFFF;
    --r-md:10px; --r-lg:16px; --r-xl:20px;
    --sh-sm:0 1px 4px rgba(0,0,0,0.06);
    --sh-md:0 4px 16px rgba(0,0,0,0.09);
}
html,body,[class*="css"],.stMarkdown,p,span,div {
    font-family:'Inter',sans-serif !important;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1.2rem 2rem 3rem !important;max-width:1360px !important;}

/* ── Header ── */
.med-header{
    background:linear-gradient(135deg,#0A2E52 0%,#0F4C7A 45%,#1B6CA8 100%);
    border-radius:var(--r-xl);padding:26px 34px;margin-bottom:22px;
    display:flex;align-items:center;gap:20px;position:relative;overflow:hidden;
}
.med-header::after{
    content:'';position:absolute;right:-50px;top:-50px;
    width:220px;height:220px;border-radius:50%;
    background:rgba(255,255,255,0.04);
}
.med-header-icon{font-size:2.5rem;z-index:1;}
.med-header h1{
    font-family:'Sora',sans-serif !important;font-size:1.7rem;font-weight:700;
    color:white;margin:0;letter-spacing:-0.5px;z-index:1;
}
.med-header p{font-size:0.82rem;color:rgba(255,255,255,0.7);margin:5px 0 0;z-index:1;}
.hbadges{display:flex;gap:7px;margin-top:9px;flex-wrap:wrap;}
.hbadge{
    background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
    color:white;font-size:0.7rem;font-weight:600;padding:3px 10px;
    border-radius:99px;letter-spacing:0.04em;
}

/* ── Disclaimer ── */
.disclaimer{
    display:flex;align-items:flex-start;gap:12px;
    padding:13px 18px;background:var(--amber-100);
    border:1px solid #F6B93B;border-left:4px solid var(--amber-500);
    border-radius:var(--r-md);margin-bottom:20px;
    font-size:0.83rem;color:#92400E;line-height:1.55;
}

/* ── Section label ── */
.slabel{
    font-size:0.67rem;font-weight:700;letter-spacing:0.12em;
    text-transform:uppercase;color:var(--gray-500);
    margin:20px 0 8px;display:flex;align-items:center;gap:8px;
}
.slabel::after{content:'';flex:1;height:1px;background:var(--gray-200);}

/* ── Form wrapper ── */
.fwrap{
    background:var(--white);border:1px solid var(--gray-200);
    border-radius:var(--r-xl);padding:24px 24px 8px;
    box-shadow:var(--sh-sm);margin-bottom:20px;
}
.int-labels{
    display:flex;justify-content:space-between;
    font-size:0.74rem;color:var(--gray-500);
    margin:-6px 0 10px;padding:0 2px;
}

/* ── Submit ── */
div[data-testid="stFormSubmitButton"] button{
    background:linear-gradient(135deg,#0A2E52,#1B6CA8) !important;
    color:white !important;border:none !important;
    border-radius:var(--r-md) !important;
    font-family:'Inter',sans-serif !important;
    font-weight:600 !important;font-size:0.94rem !important;
    padding:13px 28px !important;width:100% !important;
    box-shadow:0 4px 18px rgba(15,76,122,0.38) !important;
    transition:all 0.2s !important;
}
div[data-testid="stFormSubmitButton"] button:hover{
    transform:translateY(-1px) !important;
    box-shadow:0 6px 24px rgba(15,76,122,0.46) !important;
}

/* ── KPI cards ── */
.kpi{
    background:var(--white);border:1px solid var(--gray-200);
    border-radius:var(--r-lg);padding:15px 17px;box-shadow:var(--sh-sm);
}
.kpi-lbl{font-size:0.67rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.09em;color:var(--gray-500);margin-bottom:5px;}
.kpi-val{font-size:1.55rem;font-weight:700;color:var(--blue-800);line-height:1.1;}
.kpi-sub{font-size:0.73rem;color:var(--gray-500);margin-top:3px;}
.kpi.danger .kpi-val{color:var(--red-600);}

/* ── Urgence ── */
.urgent{
    display:flex;align-items:center;gap:16px;
    padding:16px 20px;background:var(--red-100);
    border:1.5px solid #FCA5A5;border-left:5px solid var(--red-600);
    border-radius:var(--r-lg);margin-bottom:18px;
}
.urgent h3{font-size:0.94rem;font-weight:700;color:var(--red-600);margin:0 0 3px;}
.urgent p{font-size:0.82rem;color:#7F1D1D;margin:0;line-height:1.5;}

/* ── Spec cards ── */
.spec-card{
    background:var(--white);border:1px solid var(--gray-200);
    border-radius:var(--r-lg);padding:15px 17px;margin-bottom:9px;
    display:flex;align-items:center;gap:13px;box-shadow:var(--sh-sm);
    transition:box-shadow 0.15s,transform 0.15s;
}
.spec-card:hover{box-shadow:var(--sh-md);transform:translateY(-1px);}
.spec-card.r1{border-left:4px solid #D4AF37;}
.spec-card.r2{border-left:4px solid #A8A9AD;}
.spec-card.r3{border-left:4px solid #CD7F32;}
.sp-rank{font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:700;
    min-width:24px;text-align:center;}
.r1 .sp-rank{color:#D4AF37;} .r2 .sp-rank{color:#A8A9AD;} .r3 .sp-rank{color:#CD7F32;}
.sp-icon{font-size:1.5rem;flex-shrink:0;}
.sp-info{flex:1;min-width:0;}
.sp-name{font-weight:600;font-size:0.93rem;color:var(--gray-950);}
.sp-sub{font-size:0.73rem;color:var(--gray-500);}
.sp-pct{font-size:1.05rem;font-weight:700;color:var(--blue-600);white-space:nowrap;}

/* ── Chart card ── */
.ccrd{
    background:var(--white);border:1px solid var(--gray-200);
    border-radius:var(--r-xl);padding:4px 2px 6px;box-shadow:var(--sh-sm);
}

/* ── GenAI blocks ── */
.gblk{
    background:var(--white);border:1px solid var(--gray-200);
    border-radius:var(--r-lg);padding:17px 20px;margin-bottom:12px;
    box-shadow:var(--sh-sm);
}
.gblk-ttl{
    font-size:0.67rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.1em;color:var(--blue-600);margin-bottom:9px;
    display:flex;align-items:center;gap:7px;
}
.gblk-body{font-size:0.89rem;color:var(--gray-700);line-height:1.72;}

/* ── RAG items ── */
.ritem{
    background:var(--gray-50);border:1px solid var(--gray-200);
    border-left:3px solid var(--blue-400);border-radius:var(--r-md);
    padding:10px 13px;margin-bottom:7px;
    font-size:0.81rem;color:var(--gray-700);line-height:1.5;
}
.ritem-hd{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;}
.rbadge{background:var(--blue-50);color:var(--blue-800);font-size:0.71rem;
    font-weight:600;padding:2px 8px;border-radius:99px;}
.rsim{font-size:0.71rem;color:var(--gray-500);font-weight:500;}

/* ── F4.1 ── */
.f41{
    display:flex;align-items:center;gap:10px;padding:10px 15px;
    background:var(--blue-50);border:1px solid #BFDBFE;
    border-radius:var(--r-md);margin-bottom:14px;
    font-size:0.82rem;color:var(--blue-800);
}

/* ── Chips ── */
.chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:11px;}
.chip{font-size:0.73rem;font-weight:500;padding:4px 11px;
    border-radius:99px;border:1px solid;}
.chip-b{background:var(--blue-50);color:var(--blue-800);border-color:var(--blue-100);}

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
    background:var(--gray-50) !important;
    border-right:1px solid var(--gray-200) !important;
}
.sb-brand{display:flex;align-items:center;gap:9px;
    padding:6px 0 16px;border-bottom:1px solid var(--gray-200);margin-bottom:16px;}
.sb-name{font-family:'Sora',sans-serif;font-weight:700;
    font-size:1.02rem;color:var(--blue-800);}
.sb-sec{font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
    text-transform:uppercase;color:var(--gray-500);margin:16px 0 7px;}
.sb-box{background:var(--blue-50);border-radius:var(--r-md);
    padding:11px 13px;font-size:0.79rem;color:var(--blue-800);line-height:1.65;}
.sb-box code{background:rgba(27,108,168,0.12);padding:1px 5px;
    border-radius:4px;font-size:0.73rem;}

/* ── Streamlit overrides ── */
.stSlider>div{padding:0 !important;}
div[data-testid="stTextArea"] textarea{
    border-radius:var(--r-md) !important;font-size:0.89rem !important;
    border-color:var(--gray-200) !important;}
div[data-testid="stSelectbox"]>div,
div[data-testid="stTextInput"]>div>div{border-radius:var(--r-md) !important;}
[data-testid="stExpander"]{
    border:1px solid var(--gray-200) !important;border-radius:var(--r-lg) !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
RED_FLAG_OPTIONS = [
    "douleur thoracique intense","essoufflement / dyspnée aiguë",
    "gêne respiratoire sévère","hémoptysie (sang dans les crachats)",
    "syncope / perte de connaissance","palpitations + malaise",
    "fièvre > 39°C persistante","altération de la conscience",
    "déficit moteur ou sensoriel brutal","trouble soudain de la parole",
    "céphalée en coup de tonnerre","douleur abdominale + état de choc",
    "sang dans les selles / hématémèse","vomissements incoercibles",
    "surdité brutale","baisse visuelle brutale",
    "douleur lombaire avec déficit moteur",
    "œdème de Quincke / choc anaphylactique",
    "grossesse + douleurs abdominales intenses",
    "douleur articulaire + fièvre élevée",
]

ICONS = {
    "Cardiologie":"🫀","Pneumologie":"🫁","Neurologie":"🧠",
    "Gastroenterologie":"🫃","Dermatologie":"🩹","ORL":"👂",
    "Ophtalmologie":"👁️","Endocrinologie":"⚗️","Rhumatologie":"🦴",
    "Urologie":"💧","Gynecologie":"🌸","Nephrologie":"🫘",
    "Infectiologie":"🦠","Psychiatrie":"🧩","Medecine interne":"🔬",
    "Medecine urgence":"🚨","Hematologie":"🩸","Allergologie":"🌿",
    "Orthopedie":"🦴","Pediatrie":"👶","Oncologie":"🔭",
}

DUREE_OPTIONS = ["moins de 24h","1-3 jours","1 semaine","chronique"]

# Palette graphiques
C_BLUE   = "#1B6CA8"
C_TEAL   = "#0D9488"
C_AMBER  = "#D97706"
C_PURPLE = "#7C3AED"
C_GREEN  = "#059669"
C_RED    = "#DC2626"
PALETTE  = [C_BLUE, C_TEAL, C_AMBER]

PLOTLY_CONFIG = dict(displayModeBar=False, responsive=True)
LAYOUT_BASE   = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#3A3A3C"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Composants HTML
# ─────────────────────────────────────────────────────────────────────────────
def spec_card(rank, name, score):
    rc  = {1:"r1",2:"r2",3:"r3"}.get(rank,"")
    ico = ICONS.get(name,"🏥")
    return f"""
<div class="spec-card {rc}">
  <div class="sp-rank">{rank}</div>
  <div class="sp-icon">{ico}</div>
  <div class="sp-info">
    <div class="sp-name">{name}</div>
    <div class="sp-sub">Spécialité recommandée</div>
  </div>
  <div class="sp-pct">{round(score*100,1)}%</div>
</div>"""

def rag_card(specialite, sim, text):
    ico = ICONS.get(specialite,"🏥")
    txt = text[:280]+"…" if len(text)>280 else text
    return f"""
<div class="ritem">
  <div class="ritem-hd">
    <span class="rbadge">{ico} {specialite}</span>
    <span class="rsim">sim. {sim:.3f}</span>
  </div>
  <div>{txt}</div>
</div>"""

def genai_card(icon, title, text):
    body = text.replace("\n\n","</p><p>").replace("\n","<br>")
    return f"""
<div class="gblk">
  <div class="gblk-ttl">{icon}&nbsp;{title}</div>
  <div class="gblk-body"><p>{body}</p></div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# 📡 GRAPHIQUE 1 — Radar chart (top-3 sur 3 axes de score)
# ─────────────────────────────────────────────────────────────────────────────
def make_radar(df_scores: pd.DataFrame, top3: pd.DataFrame) -> go.Figure:
    """Compare les 3 spécialités du top-3 sur les axes Symptômes / Indications / Numérique."""
    cats = ["Symptômes", "Indications", "Score numérique"]

    fig = go.Figure()
    for i, (_, row) in enumerate(top3.iterrows()):
        name  = row["Specialite"]
        match = df_scores[df_scores["Specialite"] == name]
        if match.empty:
            continue
        s = match.iloc[0]
        vals = [
            float(s.get("ScoreSymptomes",  0)),
            float(s.get("ScoreIndications",0)),
            float(s.get("ScoreNumerique",  0)),
        ]
        color = PALETTE[i % len(PALETTE)]
        icon  = ICONS.get(name,"🏥")

        # Convert hex color to rgba with alpha
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        fillcolor = f"rgba({r},{g},{b},0.133)"

        fig.add_trace(go.Scatterpolar(
            r     = vals + [vals[0]],
            theta = cats + [cats[0]],
            fill  = "toself",
            fillcolor = fillcolor,
            line  = dict(color=color, width=2.5),
            name  = f"{icon} {name}",
            hovertemplate = "<b>%{theta}</b><br>Score : %{r:.3f}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0,1],
                tickfont=dict(size=9, color="#636366"),
                gridcolor="#E5E5EA", linecolor="#E5E5EA",
                tickformat=".1f", nticks=5,
            ),
            angularaxis=dict(
                tickfont=dict(size=11.5, color="#1C1C1E"),
                gridcolor="#E5E5EA", linecolor="#E5E5EA",
            ),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.28,
            xanchor="center", x=0.5, font=dict(size=10.5),
        ),
        height=330,
        margin=dict(l=40, r=40, t=16, b=64),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 📊 GRAPHIQUE 2 — Barres empilées horizontales (top-8 spécialités)
# ─────────────────────────────────────────────────────────────────────────────
def make_bars(df_scores: pd.DataFrame, n: int = 8) -> go.Figure:
    """Top-N spécialités, score décomposé en 3 composantes empilées."""
    df = (
        df_scores
        .drop_duplicates(subset="Specialite", keep="first")
        .head(n)
        .sort_values("ScoreGlobal", ascending=True)
        .reset_index(drop=True)
    )

    labels = [f"{ICONS.get(s,'🏥')} {s}" for s in df["Specialite"]]

    fig = go.Figure()

    traces = [
        ("Symptômes ×0.6",   "ScoreSymptomes",  0.60, C_BLUE),
        ("Indications ×0.3", "ScoreIndications", 0.30, C_TEAL),
        ("Numérique ×0.1",   "ScoreNumerique",   0.10, C_AMBER),
    ]
    for label, col, weight, color in traces:
        x_vals = (df[col] * weight).round(4)
        fig.add_trace(go.Bar(
            y=labels, x=x_vals, orientation="h",
            name=label, marker_color=color,
            hovertemplate=f"<b>%{{y}}</b><br>{label} : %{{x:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        barmode="stack",
        xaxis=dict(
            title=dict(text="Score global pondéré", font=dict(size=10.5)),
            tickformat=".2f", gridcolor="#F2F2F7", range=[0, 1],
            zeroline=False,
        ),
        yaxis=dict(tickfont=dict(size=10.5), gridcolor="rgba(0,0,0,0)"),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.32,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
        height=max(240, n * 36),
        margin=dict(l=10, r=20, t=14, b=72),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 🔍 GRAPHIQUE 3 — Lollipop chart (similarité passages RAG)
# ─────────────────────────────────────────────────────────────────────────────
def make_lollipop(retrieved: list) -> go.Figure | None:
    """Visualise la similarité cosinus de chaque passage RAG récupéré."""
    if not retrieved:
        return None

    labels  = [f"{ICONS.get(p.specialite,'🏥')} {p.specialite}" for p in retrieved]
    scores  = [round(p.score, 4) for p in retrieved]
    med_ids = [p.med_id for p in retrieved]

    # Couleur selon score
    colors = [C_GREEN if s >= 0.7 else C_BLUE if s >= 0.5 else C_AMBER for s in scores]

    fig = go.Figure()

    # Tiges
    for i, (sc, col) in enumerate(zip(scores, colors)):
        fig.add_trace(go.Scatter(
            x=[0, sc], y=[i, i], mode="lines",
            line=dict(color=col, width=2.2),
            showlegend=False, hoverinfo="skip",
        ))

    # Cercles + labels de valeur
    fig.add_trace(go.Scatter(
        x=scores, y=list(range(len(labels))),
        mode="markers+text",
        marker=dict(size=14, color=colors,
                    line=dict(color="white", width=2.5)),
        text=[f"  {s:.2f}" for s in scores],
        textposition="middle right",
        textfont=dict(size=10, color="#1C1C1E"),
        customdata=med_ids,
        hovertemplate="<b>%{y}</b><br>ID : %{customdata}<br>Sim. : %{x:.3f}<extra></extra>",
        showlegend=False,
    ))

    # Légende manuelle (shapes de couleur)
    for color, label, threshold in [
        (C_GREEN, "Haute (≥ 0.70)", None),
        (C_BLUE,  "Moyenne (0.50–0.69)", None),
        (C_AMBER, "Basse (< 0.50)", None),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        xaxis=dict(
            title=dict(text="Similarité cosinus", font=dict(size=10.5)),
            range=[0, 1.18], tickformat=".1f",
            gridcolor="#F2F2F7", zeroline=False,
        ),
        yaxis=dict(
            tickvals=list(range(len(labels))),
            ticktext=labels,
            tickfont=dict(size=10.5),
            gridcolor="rgba(0,0,0,0)",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.30,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
        height=max(200, len(retrieved) * 48),
        margin=dict(l=10, r=64, t=14, b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <span style="font-size:1.55rem">🩺</span>
      <span class="sb-name">MedOrient</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Modèle IA</div>', unsafe_allow_html=True)
    model_name = os.getenv("GENAI_MODEL", "gemini-2.5-flash")
    st.markdown(f"""
    <div class="sb-box">
      🤖 <strong>LLM</strong> : <code>{model_name}</code><br>
      📐 <strong>Embeddings</strong> : <code>all-MiniLM-L6-v2</code><br>
      🗂️ <strong>Index</strong> : <code>FAISS cosine</code>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">Paramètres RAG</div>', unsafe_allow_html=True)
    retrieve_k    = st.slider("Passages récupérés (k)", 1, 10, 5,
                              help="Passages du référentiel injectés dans le prompt Gemini")
    force_rebuild = st.checkbox("🔄 Reconstruire l'index FAISS",
                                help="Ignore le cache et reconstruit l'index vectoriel")

    st.markdown('<div class="sb-sec">Pondération</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sympt.", "60%"); c2.metric("Indic.", "30%"); c3.metric("Num.", "10%")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.76rem;color:#636366;line-height:1.75">
      ⚠️ Outil d'orientation uniquement.<br>
      🆘 Urgence : <strong>15</strong> · <strong>112</strong> · <strong>18</strong>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="med-header">
  <div class="med-header-icon">🩺</div>
  <div>
    <h1>MedOrient</h1>
    <p>Agent d'Orientation Médicale Intelligent — RAG · SBERT · Gemini 2.5 Flash</p>
    <div class="hbadges">
      <span class="hbadge">SBERT</span>
      <span class="hbadge">FAISS</span>
      <span class="hbadge">Gemini 2.5 Flash</span>
      <span class="hbadge">F4.1 Query Expansion</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
  ⚠️&nbsp;<span>
    <strong>Avertissement :</strong> Outil d'<em>orientation</em> uniquement — pas de diagnostic.
    Consultez un professionnel de santé. Urgence : <strong>15</strong> · <strong>112</strong> · <strong>18</strong>
  </span>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Formulaire
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="fwrap">', unsafe_allow_html=True)

with st.form("input_form"):
    st.markdown('<div class="slabel">📝 Description des symptômes</div>', unsafe_allow_html=True)
    description = st.text_area(
        "Symptômes", height=118, label_visibility="collapsed",
        placeholder="Décrivez vos symptômes en détail — ex : douleur thoracique depuis 2 jours, irradiant vers l'épaule gauche, accompagnée d'essoufflement au moindre effort…",
    )

    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown('<div class="slabel">📍 Localisation</div>', unsafe_allow_html=True)
        localisation = st.text_input("Localisation", value="",
            placeholder="Ex : poitrine, tête…", label_visibility="collapsed")
        st.markdown('<div class="slabel">⏱️ Durée</div>', unsafe_allow_html=True)
        duree = st.selectbox("Durée", DUREE_OPTIONS, label_visibility="collapsed")
    with c2:
        st.markdown('<div class="slabel">🔥 Intensité (1–5)</div>', unsafe_allow_html=True)
        intensite = st.slider("Intensité", 1, 5, 3, label_visibility="collapsed")
        st.markdown("""
        <div class="int-labels">
          <span>1 · Légère</span><span>3 · Modérée</span><span>5 · Sévère</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="slabel">🎂 Tranche d\'âge</div>', unsafe_allow_html=True)
        age_tranche = st.selectbox("Âge",
            ["Enfant (< 15 ans)","Adulte jeune (15–40 ans)","Adulte (40–65 ans)","Senior (> 65 ans)"],
            index=1, label_visibility="collapsed")

    st.markdown('<div class="slabel">🚨 Signaux d\'alerte</div>', unsafe_allow_html=True)
    rc1, rc2 = st.columns([2.2, 1])
    with rc1:
        red_flags = st.multiselect("Red flags", RED_FLAG_OPTIONS,
            label_visibility="collapsed",
            placeholder="Sélectionnez les signaux d'alerte présents…")
    with rc2:
        red_flag_other = st.text_input("Autre", placeholder="Autre signal…",
            label_visibility="collapsed")

    submitted = st.form_submit_button("🔍  Analyser mes symptômes", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Résultats
# ─────────────────────────────────────────────────────────────────────────────
if submitted:
    if not description.strip():
        st.warning("⚠️ Veuillez décrire vos symptômes avant de lancer l'analyse.")
        st.stop()

    all_rf = red_flags + ([red_flag_other.strip()] if red_flag_other.strip() else [])
    answers = {
        "description": description, "intensite": intensite,
        "duree": duree, "localisation": localisation or "general",
        "red_flags": all_rf, "age_tranche": age_tranche,
    }

    with st.spinner("⏳ Analyse en cours — embeddings · RAG · Gemini…"):
        try:
            out = run_pipeline_with_generation(
                answers, retrieve_k=retrieve_k, retrieve_force_rebuild=force_rebuild)
        except Exception as e:
            st.error(f"❌ Erreur : {e}"); st.stop()

    # ── Urgence ──────────────────────────────────────────────────────────────
    if out.get("red_flags_detected"):
        flags_str = " · ".join(out["red_flags_hits"])
        st.markdown(f"""
        <div class="urgent">
          <span style="font-size:2rem;flex-shrink:0">🚨</span>
          <div>
            <h3>Signaux d'urgence détectés — consultez immédiatement</h3>
            <p>Appelez le <strong>15 (SAMU)</strong> ou le <strong>112</strong>.
            Signaux : <em>{flags_str}</em></p>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── F4.1 ─────────────────────────────────────────────────────────────────
    if out.get("description_expanded"):
        st.markdown("""
        <div class="f41">
          ✨ <strong>F4.1 activé</strong> — description enrichie par Gemini
          (texte trop court détecté) pour améliorer la précision des embeddings.
        </div>""", unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    top_spec  = out["top3"].iloc[0]["Specialite"] if len(out["top3"]) else "—"
    top_score = out["top3"].iloc[0]["ScoreGlobal"] if len(out["top3"]) else 0
    n_flags   = len(out["red_flags_hits"]) if out.get("red_flags_detected") else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-lbl">Spécialité n°1</div>
          <div class="kpi-val" style="font-size:1.05rem">
            {ICONS.get(top_spec,'🏥')} {top_spec}</div>
          <div class="kpi-sub">Score {round(top_score*100,1)}%</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-lbl">Score numérique</div>
          <div class="kpi-val">{out['numeric_score']:.2f}</div>
          <div class="kpi-sub">Intensité · Durée · Red flags</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-lbl">Passages RAG</div>
          <div class="kpi-val">{len(out['retrieved'])}</div>
          <div class="kpi-sub">Extraits du référentiel</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        cls = "kpi danger" if n_flags else "kpi"
        st.markdown(f"""
        <div class="{cls}">
          <div class="kpi-lbl">Red flags</div>
          <div class="kpi-val">{n_flags}</div>
          <div class="kpi-sub">{"⚠️ Urgence" if n_flags else "Aucun signal"}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:26px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # LIGNE 1 — Top-3 (gauche) + Radar chart (droite)
    # ═════════════════════════════════════════════════════════════════════════
    col_spec, col_radar = st.columns([1, 1.1], gap="large")

    with col_spec:
        st.markdown('<div class="slabel">🏆 Spécialités recommandées</div>', unsafe_allow_html=True)
        for i, row in out["top3"].iterrows():
            st.markdown(spec_card(i+1, row["Specialite"], row["ScoreGlobal"]),
                        unsafe_allow_html=True)
        # Chips organes
        organs, seen = [], set()
        for org_str in out["top3"].get("Organes", pd.Series()).tolist():
            for o in str(org_str).split(","):
                o = o.strip()
                if o and o not in seen:
                    seen.add(o); organs.append(o)
        if organs:
            chips = "".join(f'<span class="chip chip-b">{o}</span>' for o in organs[:8])
            st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    with col_radar:
        st.markdown('<div class="slabel">📡 Radar — comparaison des 3 spécialités</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ccrd">', unsafe_allow_html=True)
        st.plotly_chart(make_radar(out["scores"], out["top3"]),
                        use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # LIGNE 2 — Barres empilées (gauche) + Lollipop RAG (droite)
    # ═════════════════════════════════════════════════════════════════════════
    col_bar, col_lolly = st.columns([1.2, 1], gap="large")

    with col_bar:
        st.markdown('<div class="slabel">📊 Top-8 spécialités — décomposition des scores</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ccrd">', unsafe_allow_html=True)
        st.plotly_chart(make_bars(out["scores"], n=8),
                        use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_lolly:
        st.markdown('<div class="slabel">🔍 Similarité des passages RAG récupérés</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ccrd">', unsafe_allow_html=True)
        fig_l = make_lollipop(out["retrieved"])
        if fig_l:
            st.plotly_chart(fig_l, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Aucun passage RAG récupéré.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:26px'></div>", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    # LIGNE 3 — Textes GenAI (gauche) + Passages RAG (droite)
    # ═════════════════════════════════════════════════════════════════════════
    col_ai, col_rag = st.columns([1.1, 1], gap="large")

    with col_ai:
        st.markdown('<div class="slabel">🤖 Analyse générée par Gemini</div>',
                    unsafe_allow_html=True)
        if out.get("genai_text"):
            st.markdown(genai_card("💬","Orientation & Explication", out["genai_text"]),
                        unsafe_allow_html=True)
        if out.get("plan_text"):
            st.markdown(genai_card("📋","Plan d'action recommandé", out["plan_text"]),
                        unsafe_allow_html=True)
        if out.get("bio_text"):
            st.markdown(genai_card("📄","Synthèse patient", out["bio_text"]),
                        unsafe_allow_html=True)

    with col_rag:
        st.markdown('<div class="slabel">📂 Passages du référentiel (RAG)</div>',
                    unsafe_allow_html=True)
        if out["retrieved"]:
            for p in out["retrieved"]:
                st.markdown(rag_card(p.specialite, p.score, p.text),
                            unsafe_allow_html=True)
        else:
            st.info("Aucun passage récupéré.")

    # ── Tableau détaillé ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
    with st.expander("📋 Tableau détaillé des scores — toutes spécialités", expanded=False):
        cols_to_show = [c for c in ["Specialite","ScoreGlobal","ScoreSymptomes",
                                    "ScoreIndications","ScoreNumerique"]
                        if c in out["scores"].columns]
        df_disp = out["scores"][cols_to_show].head(15).copy()
        for col in df_disp.select_dtypes("float").columns:
            df_disp[col] = df_disp[col].map(lambda x: f"{x:.4f}")
        st.dataframe(df_disp, use_container_width=True, hide_index=True)
