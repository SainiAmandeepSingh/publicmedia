# app/app.py
# NPO Start — Public Values Recommender System
# Utrecht University | INFOMPPM | Assignment 2
# Run with: streamlit run app/app.py

import sys, os, json
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.synthetic_data import generate_catalogue, generate_observation_sample, parse_genres
from src.user_profiles import generate_users, apply_user_preferences, NPO_PERSONAS
from src.scoring import build_feature_matrix, score_items_for_user, DEFAULT_POPULARITY_BIAS
from src.diversity import rerank_for_diversity, compute_ils
from src.fairness import (
    compute_cat_share, compute_rec_share, compute_exposure_gap,
    rerank_for_fairness, fairness_correction,
)
from src.transparency import get_primary_reason, get_feature_details

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPO Start — Public Values Recommender",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NPO Design System ─────────────────────────────────────────────────────────
# Colours extracted directly from npo.nl/start CSS computed styles
NPO_BG_DEEP    = "#0A1931"   # rgb(10,25,49)   — page background, navbar
NPO_BG_CARD    = "#081426"   # rgb(8,20,38)    — card backgrounds
NPO_BG_MID     = "#1F3353"   # rgb(31,51,83)   — card hover, secondary bg
NPO_BG_BORDER  = "#293D5D"   # rgb(41,61,93)   — borders, dividers
NPO_ORANGE     = "#F56A00"   # rgb(245,106,0)  — brand accent
NPO_ORANGE_DIM = "#C45400"   # darker orange   — hover states
NPO_WHITE      = "#FFFFFF"
NPO_WHITE_DIM  = "rgba(255,255,255,0.65)"
NPO_WHITE_SUB  = "rgba(255,255,255,0.40)"

# Broadcaster colours — distinct, accessible on dark background
BROADCASTER_COLOURS = {
    'AVROTROS': '#E05252',
    'MAX':      '#E8953A',
    'KRO-NCRV': '#3AAEA0',
    'VPRO':     '#5B8FCC',
    'NTR':      '#8A6CC7',
    'EO':       '#5BBF8A',
    'BNNVARA':  '#D4A843',
}

# ── Global CSS — NPO Start look and feel ─────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Google Fonts fallback (NPO uses proprietary fonts) ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  /* ── Page background ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stApp"], .main {{
    background-color: {NPO_BG_DEEP} !important;
    color: {NPO_WHITE} !important;
    font-family: 'Inter', 'NPO Sans', sans-serif !important;
  }}

  /* ── Main content area ── */
  [data-testid="stMainBlockContainer"],
  [data-testid="block-container"] {{
    background-color: {NPO_BG_DEEP} !important;
    padding-top: 1rem !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background-color: {NPO_BG_CARD} !important;
    border-right: 1px solid {NPO_BG_BORDER} !important;
  }}
  [data-testid="stSidebar"] * {{
    color: {NPO_WHITE} !important;
  }}
  [data-testid="stSidebar"] .stSlider > div > div > div {{
    background: {NPO_ORANGE} !important;
  }}

  /* ── Tabs ── */
  [data-testid="stTabs"] > div:first-child {{
    background: transparent !important;
    border-bottom: 2px solid {NPO_BG_BORDER} !important;
    gap: 0 !important;
  }}
  button[data-baseweb="tab"] {{
    background: transparent !important;
    color: {NPO_WHITE_DIM} !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.6rem 1.2rem !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    transition: all 0.15s ease !important;
  }}
  button[data-baseweb="tab"]:hover {{
    color: {NPO_WHITE} !important;
    background: rgba(255,255,255,0.05) !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    color: {NPO_WHITE} !important;
    border-bottom: 3px solid {NPO_ORANGE} !important;
    background: transparent !important;
  }}
  [data-testid="stTabsContent"] {{
    background: transparent !important;
    padding-top: 1.5rem !important;
  }}

  /* ── Metrics ── */
  [data-testid="metric-container"] {{
    background: {NPO_BG_CARD} !important;
    border: 1px solid {NPO_BG_BORDER} !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
  }}
  [data-testid="metric-container"] label {{
    color: {NPO_WHITE_DIM} !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {NPO_WHITE} !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
  }}

  /* ── Dataframes / tables ── */
  [data-testid="stDataFrame"] {{
    background: {NPO_BG_CARD} !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid {NPO_BG_BORDER} !important;
  }}

  /* ── Info / warning boxes ── */
  [data-testid="stInfo"] {{
    background: rgba(91, 143, 204, 0.12) !important;
    border-left: 3px solid #5B8FCC !important;
    border-radius: 0 6px 6px 0 !important;
    color: {NPO_WHITE} !important;
  }}
  [data-testid="stWarning"] {{
    background: rgba(245,106,0,0.12) !important;
    border-left: 3px solid {NPO_ORANGE} !important;
    color: {NPO_WHITE} !important;
  }}
  [data-testid="stSuccess"] {{
    background: rgba(91,191,138,0.12) !important;
    border-left: 3px solid #5BBF8A !important;
    color: {NPO_WHITE} !important;
  }}

  /* ── Dividers ── */
  hr {{
    border-color: {NPO_BG_BORDER} !important;
    opacity: 0.6 !important;
  }}

  /* ── Selects / inputs ── */
  [data-testid="stSelectbox"] > div,
  [data-testid="stMultiSelect"] > div {{
    background: {NPO_BG_MID} !important;
    border: 1px solid {NPO_BG_BORDER} !important;
    border-radius: 6px !important;
    color: {NPO_WHITE} !important;
  }}

  /* ── Toggle ── */
  [data-testid="stToggle"] span {{
    background: {NPO_BG_MID} !important;
  }}

  /* ── Caption / small text ── */
  [data-testid="stCaptionContainer"] p,
  .stCaption {{
    color: {NPO_WHITE_DIM} !important;
    font-size: 0.8rem !important;
  }}

  /* ── Section headers ── */
  h1, h2, h3 {{
    color: {NPO_WHITE} !important;
    font-family: 'Inter', 'NPO Scandia', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
  }}

  /* ── Plotly chart backgrounds ── */
  .js-plotly-plot .plotly .bg {{
    fill: transparent !important;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: {NPO_BG_DEEP}; }}
  ::-webkit-scrollbar-thumb {{ background: {NPO_BG_BORDER}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {NPO_ORANGE}; }}
</style>
""", unsafe_allow_html=True)

# ── Data loading — real CSV first, synthetic fallback ─────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

@st.cache_data
def load_all():
    cat_path = DATA_DIR / "catalogue.csv"
    rs_path  = DATA_DIR / "rec_share.json"

    if cat_path.exists():
        cat = pd.read_csv(cat_path)
        cat['genres'] = cat['genres'].apply(parse_genres)
        if 'item_id' not in cat.columns and 'slug' in cat.columns:
            cat = cat.rename(columns={'slug': 'item_id'})
        rec_share_bl = json.loads(rs_path.read_text()) if rs_path.exists() \
                       else compute_rec_share(cat)
        data_source = f"🟢 Real data — {len(cat)} NPO series from data/processed/"
    else:
        cat = generate_catalogue(n_items=300, seed=42)
        obs = generate_observation_sample(cat, n_sessions=200, seed=42)
        rec_share_bl = compute_rec_share(obs)
        data_source = f"🟡 Synthetic data — {len(cat)} items  (run `python src/data_loader.py` for real data)"

    users      = generate_users(cat, n_users=30, seed=42)
    cat_share  = compute_cat_share(cat)
    fm, ids, _ = build_feature_matrix(cat)
    return cat, users, cat_share, rec_share_bl, fm, ids, data_source

cat, users_df, cat_share, rec_share_baseline, feature_matrix, item_ids, data_source = load_all()

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(user_profile, lambda_weight, diversity_factor, top_n):
    scored = score_items_for_user(cat, user_profile, feature_matrix, item_ids, DEFAULT_POPULARITY_BIAS)
    scored = apply_user_preferences(scored, user_profile)
    mn, mx = scored['base_score'].min(), scored['base_score'].max()
    scored['base_score'] = (scored['base_score'] - mn) / (mx - mn + 1e-9)
    scored['current_score'] = scored['base_score']
    diverse = rerank_for_diversity(scored.head(top_n * 6), top_n=top_n * 2, diversity_factor=diversity_factor)
    diverse['current_score'] = diverse['base_score']
    final = rerank_for_fairness(
        diverse, cat_share, rec_share_baseline, lambda_weight=lambda_weight
    ).head(top_n)
    return scored, final

# ── Plotly theme helper ───────────────────────────────────────────────────────
def npo_chart_layout(**kwargs):
    base = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=NPO_WHITE, family='Inter, sans-serif', size=12),
        xaxis=dict(
            gridcolor=NPO_BG_BORDER, linecolor=NPO_BG_BORDER,
            tickfont=dict(color=NPO_WHITE_DIM), title_font=dict(color=NPO_WHITE_DIM),
        ),
        yaxis=dict(
            gridcolor=NPO_BG_BORDER, linecolor=NPO_BG_BORDER,
            tickfont=dict(color=NPO_WHITE_DIM), title_font=dict(color=NPO_WHITE_DIM),
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)', font=dict(color=NPO_WHITE_DIM),
            orientation='h', y=-0.3,
        ),
        margin=dict(t=10, b=10, l=5, r=5),
    )
    base.update(kwargs)
    return base

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # NPO logo area
    st.markdown(f"""
    <div style="padding:1rem 0 0.5rem 0;display:flex;align-items:center;gap:10px">
      <div style="background:{NPO_ORANGE};width:36px;height:36px;border-radius:6px;
                  display:flex;align-items:center;justify-content:center;
                  font-weight:900;font-size:14px;color:white;flex-shrink:0">NPO</div>
      <div>
        <div style="font-weight:700;font-size:0.95rem;color:white">NPO Start</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.5)">Public Values Prototype</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(f"<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.4);margin-bottom:6px'>Viewer Persona</p>", unsafe_allow_html=True)
    persona = st.selectbox(
        "persona", list(NPO_PERSONAS.keys()),
        format_func=lambda x: x.title(),
        label_visibility="collapsed",
    )
    st.markdown(f"<p style='font-size:0.78rem;color:rgba(255,255,255,0.5);margin-top:-6px;font-style:italic'>{NPO_PERSONAS[persona]['description']}</p>", unsafe_allow_html=True)

    match = users_df[users_df['persona'] == persona]
    user_profile = match.iloc[0].to_dict() if not match.empty else {
        'user_id': 'guest', 'persona': persona,
        'preferred_genres':     NPO_PERSONAS[persona]['preferred_genres'],
        'broadcaster_affinity': NPO_PERSONAS[persona]['broadcaster_affinity'],
        'genre_weights':        NPO_PERSONAS[persona]['genre_weights'],
        'watch_history': [], 'lambda_preference': 0.5,
        'diversity_preference': 0.4, 'show_explanations': True,
    }

    st.divider()
    st.markdown(f"<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.4);margin-bottom:4px'>⚖️ Fairness Weight (λ)</p>", unsafe_allow_html=True)
    lambda_val = st.slider(
        "lambda", min_value=0.10, max_value=1.00,
        value=float(user_profile.get('lambda_preference', 0.5)), step=0.05,
        label_visibility="collapsed",
        help="Higher λ = stronger boost for underexposed broadcasters. Min 0.10 = Mediawet 2008 floor."
    )
    st.markdown(f"<p style='font-size:0.72rem;color:rgba(255,255,255,0.4);margin-top:-4px'>Min 0.10 &nbsp;·&nbsp; Mediawet 2008 floor</p>", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.4);margin-bottom:4px'>🌍 Diversity Strength</p>", unsafe_allow_html=True)
    diversity_val = st.slider(
        "diversity", min_value=0.0, max_value=1.0,
        value=float(user_profile.get('diversity_preference', 0.4)), step=0.05,
        label_visibility="collapsed",
        help="Padma Dhuney — ILS reduction via greedy diversity re-ranking."
    )

    top_n = st.slider("Recommendations", 6, 12, 9, step=3)

    st.divider()
    st.markdown(f"<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:rgba(255,255,255,0.4);margin-bottom:4px'>🎭 Genre Preferences</p>", unsafe_allow_html=True)
    all_genres   = sorted(set(g for gs in cat['genres'] for g in gs))
    pref_default = [g for g in user_profile.get('preferred_genres', []) if g in all_genres]
    selected_genres = st.multiselect(
        "genres", all_genres, default=pref_default,
        label_visibility="collapsed",
        help="Kiron Putman — explicit genre preferences override behavioural data."
    )
    if selected_genres:
        user_profile['preferred_genres'] = selected_genres
        for g in selected_genres:
            user_profile.setdefault('genre_weights', {})[g] = max(
                user_profile.get('genre_weights', {}).get(g, 0.0), 0.6)

    st.divider()
    show_explanations = st.toggle("Explanation labels", value=True,
                                  help="Lisa Wang — transparency labels on each card.")
    show_scores = st.toggle("Score breakdown", value=False)

# ── Run pipeline ──────────────────────────────────────────────────────────────
scored_df, final_df = run_pipeline(user_profile, lambda_val, diversity_val, top_n)
baseline_top = scored_df.head(top_n).copy()
baseline_top['fairness_boosted'] = False

eg_before  = compute_exposure_gap(cat_share, rec_share_baseline)
eg_after   = compute_exposure_gap(cat_share, compute_rec_share(final_df))
eg_improve = (eg_before - eg_after) / eg_before * 100 if eg_before > 0 else 0
ils_before = compute_ils(baseline_top.to_dict('records'))
ils_after  = compute_ils(final_df.to_dict('records'))

# ── Page header — NPO-style top bar ──────────────────────────────────────────
st.markdown(f"""
<div style="background:{NPO_BG_CARD};border-bottom:1px solid {NPO_BG_BORDER};
            padding:0.75rem 1.5rem;margin:-1rem -1rem 1.5rem -1rem;
            display:flex;align-items:center;justify-content:space-between;">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="background:{NPO_ORANGE};padding:4px 10px;border-radius:5px;
                font-weight:900;font-size:13px;color:white;letter-spacing:0.02em">NPO</div>
    <span style="color:{NPO_WHITE};font-size:1rem;font-weight:600">Start</span>
    <span style="color:{NPO_WHITE_SUB};font-size:0.8rem">·</span>
    <span style="color:{NPO_WHITE_SUB};font-size:0.8rem">Public Values Recommender</span>
  </div>
  <span style="font-size:0.75rem;color:{NPO_WHITE_SUB}">{data_source}</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_recs, tab_fair, tab_div, tab_profile, tab_about = st.tabs([
    "Aanbevolen voor jou",
    "⚖️ Fairness",
    "🌍 Diversity",
    "👤 Mijn Profiel",
    "ℹ️ Over",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Aanbevolen voor jou
# ══════════════════════════════════════════════════════════════════════════════
with tab_recs:
    # KPI row — NPO-style metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposure Gap — baseline", f"{eg_before:.3f}",
              help="EG before fairness re-ranking — from NPO Start observation data")
    c2.metric("Exposure Gap — na herranking", f"{eg_after:.3f}",
              delta=f"{eg_after - eg_before:+.3f}", delta_color="inverse")
    c3.metric("EG verbetering", f"{eg_improve:.0f}%")
    c4.metric("ILS reductie", f"{ils_before - ils_after:.3f}",
              help="Diversity improvement — lower ILS = more genre variety")
    st.divider()

    # ── Section header — NPO row style ───────────────────────────────────────
    col_before, col_spacer, col_after = st.columns([5, 1, 5])

    def render_card(item, user_profile, show_exp, show_score, col):
        b      = item.get('broadcaster', '')
        colour = BROADCASTER_COLOURS.get(b, '#6B7A99')
        boosted = item.get('fairness_boosted', False)
        genres  = item.get('genres') or []
        if isinstance(genres, str):
            genres = parse_genres(genres)

        genre_chips = "".join([
            f'<span style="background:rgba(255,255,255,0.08);color:rgba(255,255,255,0.7);'
            f'padding:2px 8px;border-radius:4px;font-size:0.68rem;margin:1px 2px 1px 0;'
            f'display:inline-block;border:1px solid rgba(255,255,255,0.12)">{g}</span>'
            for g in genres
        ])

        boost_html = (
            f'<span style="font-size:0.65rem;color:#5BBF8A;font-weight:600;'
            f'background:rgba(91,191,138,0.15);padding:1px 6px;border-radius:3px;'
            f'border:1px solid rgba(91,191,138,0.3)">⬆ fairness boost</span>'
        ) if boosted else ''

        left_accent = f"border-left:3px solid {colour};" if True else ""

        col.markdown(f"""
<div style="background:{NPO_BG_CARD};border-radius:8px;padding:0.85rem 1rem;
            border:1px solid {NPO_BG_BORDER};{left_accent}
            margin-bottom:0.5rem;transition:background 0.15s">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
    <span style="background:{colour};color:white;font-size:0.65rem;font-weight:700;
                 padding:2px 8px;border-radius:4px;letter-spacing:0.03em">{b}</span>
    {boost_html}
  </div>
  <div style="font-size:0.9rem;font-weight:600;color:{NPO_WHITE};
              margin-bottom:6px;line-height:1.3">{item.get('title','')}</div>
  <div style="margin-top:4px">{genre_chips}</div>
</div>""", unsafe_allow_html=True)

        if show_exp:
            reason = get_primary_reason(item, user_profile)
            col.markdown(
                f'<p style="font-size:0.75rem;color:rgba(255,255,255,0.45);'
                f'margin:-8px 0 6px 4px">{reason}</p>',
                unsafe_allow_html=True
            )
        if show_score:
            details = get_feature_details(item, user_profile)
            with col.expander("Score breakdown"):
                for k, v in details['score_breakdown'].items():
                    st.write(f"**{k}:** {v}")

    with col_before:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.75rem">
  <div style="width:10px;height:10px;background:#E05252;border-radius:50%"></div>
  <span style="font-size:0.95rem;font-weight:700;color:{NPO_WHITE}">
    CTR-only — geen eerlijkheidscorrectie</span>
</div>
<p style="font-size:0.78rem;color:{NPO_WHITE_DIM};margin-top:-10px;margin-bottom:12px">
  EG = {eg_before:.3f} &nbsp;·&nbsp; ILS = {ils_before:.3f}</p>
""", unsafe_allow_html=True)
        for item in baseline_top.to_dict('records'):
            render_card(item, user_profile, show_explanations, show_scores, col_before)

    with col_spacer:
        st.markdown(
            f'<div style="width:1px;background:{NPO_BG_BORDER};'
            f'min-height:400px;margin:28px auto 0 auto"></div>',
            unsafe_allow_html=True
        )

    with col_after:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:0.75rem">
  <div style="width:10px;height:10px;background:#5BBF8A;border-radius:50%"></div>
  <span style="font-size:0.95rem;font-weight:700;color:{NPO_WHITE}">
    Na herranking — eerlijkheidsgewicht λ = {lambda_val:.2f}</span>
</div>
<p style="font-size:0.78rem;color:{NPO_WHITE_DIM};margin-top:-10px;margin-bottom:12px">
  EG = {eg_after:.3f} &nbsp;·&nbsp; ILS = {ils_after:.3f}</p>
""", unsafe_allow_html=True)
        for item in final_df.to_dict('records'):
            render_card(item, user_profile, show_explanations, show_scores, col_after)

    st.divider()

    # ── Broadcaster share bar ─────────────────────────────────────────────────
    st.markdown(f"<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:{NPO_WHITE_DIM};margin-bottom:4px'>Omroep verdeling na herranking</p>", unsafe_allow_html=True)
    bc_counts = final_df['broadcaster'].value_counts().reset_index()
    bc_counts.columns = ['Broadcaster', 'Count']
    bc_counts['Share (%)'] = bc_counts['Count'] / bc_counts['Count'].sum() * 100
    fig_bc = px.bar(
        bc_counts, x='Broadcaster', y='Share (%)',
        color='Broadcaster', color_discrete_map=BROADCASTER_COLOURS,
        text_auto='.1f', height=220,
    )
    fig_bc.update_layout(**npo_chart_layout(showlegend=False))
    fig_bc.update_traces(marker_line_width=0)
    st.plotly_chart(fig_bc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FAIRNESS DASHBOARD  (AmanDeep Singh)
# ══════════════════════════════════════════════════════════════════════════════
with tab_fair:
    # Section header
    st.markdown(f"""
<div style="margin-bottom:1.5rem">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
    <div style="width:3px;height:24px;background:{NPO_ORANGE};border-radius:2px"></div>
    <h2 style="margin:0;font-size:1.25rem">Fairness Dashboard — Producer-side Fairness</h2>
  </div>
  <p style="color:{NPO_WHITE_DIM};font-size:0.85rem;margin:0 0 0 13px">
    AmanDeep Singh &nbsp;·&nbsp; Exposure Gap (EG) metric &nbsp;·&nbsp; Mediawet 2008
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:{NPO_WHITE_DIM};font-size:0.9rem;margin-bottom:1rem'>"
        "EG measures how far each broadcaster's recommendation share deviates from its "
        "catalogue share. Lower EG = more equitable exposure.</p>",
        unsafe_allow_html=True
    )
    st.latex(r"EG = \frac{1}{|B|} \sum_{b \in B} \left| rec\_share(b) - cat\_share(b) \right|")
    st.divider()

    # KPI row
    m1, m2, m3 = st.columns(3)
    m1.metric("EG — baseline (CTR-only)", f"{eg_before:.3f}")
    colour_eg = "#5BBF8A" if eg_after < eg_before else "#E05252"
    m2.metric("EG — na herranking", f"{eg_after:.3f}",
              delta=f"{eg_after - eg_before:+.3f}", delta_color="inverse")
    m3.metric("EG verbetering", f"{eg_improve:.0f}%")
    st.divider()

    # Before / after charts
    bc_list  = [b for b in BROADCASTER_COLOURS if b in cat_share or b in rec_share_baseline]
    cat_vals = [cat_share.get(b, 0) * 100 for b in bc_list]
    obs_vals = [rec_share_baseline.get(b, 0) * 100 for b in bc_list]
    aft_vals = [compute_rec_share(final_df).get(b, 0) * 100 for b in bc_list]

    cl, cr = st.columns(2)
    with cl:
        st.markdown(f"""
<div style="margin-bottom:0.5rem">
  <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
               color:{NPO_WHITE_DIM}">Vóór interventie</span>
  <span style="background:#E05252;color:white;font-size:0.72rem;font-weight:700;
               padding:2px 8px;border-radius:4px;margin-left:8px">EG = {eg_before:.3f}</span>
</div>
<p style="font-size:0.78rem;color:rgba(255,255,255,0.35);margin-top:2px">
  Catalogusaandeel vs. geobserveerd aanbevelingsaandeel (CTR-geoptimaliseerde baseline)</p>
""", unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name='Catalogusaandeel', x=bc_list, y=cat_vals,
                              marker_color='#5B8FCC', marker_line_width=0))
        fig1.add_trace(go.Bar(name='Aanbevelingsaandeel (baseline)', x=bc_list, y=obs_vals,
                              marker_color='#E05252', marker_line_width=0))
        fig1.update_layout(**npo_chart_layout(barmode='group', height=300,
                                               yaxis_title='Aandeel (%)'))
        st.plotly_chart(fig1, use_container_width=True)

    with cr:
        st.markdown(f"""
<div style="margin-bottom:0.5rem">
  <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
               color:{NPO_WHITE_DIM}">Na herranking (λ = {lambda_val:.2f})</span>
  <span style="background:{colour_eg};color:white;font-size:0.72rem;font-weight:700;
               padding:2px 8px;border-radius:4px;margin-left:8px">EG = {eg_after:.3f}</span>
</div>
<p style="font-size:0.78rem;color:rgba(255,255,255,0.35);margin-top:2px">
  Catalogusaandeel vs. herrankt aanbevelingsaandeel</p>
""", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Catalogusaandeel', x=bc_list, y=cat_vals,
                              marker_color='#5B8FCC', marker_line_width=0))
        fig2.add_trace(go.Bar(name='Herrankt aanbevelingsaandeel', x=bc_list, y=aft_vals,
                              marker_color='#5BBF8A', marker_line_width=0))
        fig2.update_layout(**npo_chart_layout(barmode='group', height=300,
                                               yaxis_title='Aandeel (%)'))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # λ grid search
    st.markdown(f"""
<div style="margin-bottom:0.5rem">
  <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
               color:{NPO_WHITE_DIM}">λ Kalibratie — Grid Search</span>
</div>
<p style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-bottom:1rem">
  EG wordt berekend voor elke λ-waarde in [0,1]. De optimale λ is de laagste waarde die EG
  onder de doeldrempel brengt terwijl de relevantie acceptabel blijft — zie Sectie 3.3 van het voorstel.</p>
""", unsafe_allow_html=True)

    lambda_grid = np.arange(0.0, 1.05, 0.05)
    eg_grid = []
    for lam in lambda_grid:
        _, tmp = run_pipeline(user_profile, float(lam), diversity_val, top_n)
        eg_grid.append(compute_exposure_gap(cat_share, compute_rec_share(tmp)))

    fig_lam = go.Figure()
    fig_lam.add_trace(go.Scatter(
        x=lambda_grid, y=eg_grid, mode='lines+markers',
        line=dict(color=NPO_ORANGE, width=2.5),
        marker=dict(size=6, color=NPO_ORANGE),
        name='EG per λ',
    ))
    fig_lam.add_vline(x=lambda_val, line_dash='dash', line_color='#5BBF8A', line_width=1.5,
                      annotation_text=f"λ = {lambda_val:.2f}",
                      annotation_font=dict(color='#5BBF8A', size=11))
    fig_lam.add_hline(y=eg_before, line_dash='dot', line_color='#E05252', line_width=1,
                      annotation_text="Baseline EG",
                      annotation_font=dict(color='#E05252', size=10))
    fig_lam.add_hline(y=0.05, line_dash='dot', line_color=NPO_WHITE_DIM, line_width=1,
                      annotation_text="Doeldrempel EG = 0.05",
                      annotation_font=dict(color=NPO_WHITE_DIM, size=10))
    fig_lam.update_layout(**npo_chart_layout(
        height=300, xaxis_title='λ (eerlijkheidsgewicht)', yaxis_title='Exposure Gap (EG)'
    ))
    st.plotly_chart(fig_lam, use_container_width=True)
    st.divider()

    # Fairness correction table
    st.markdown(f"""
<div style="margin-bottom:0.5rem">
  <span style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;
               color:{NPO_WHITE_DIM}">Eerlijkheidscorrectie per omroep</span>
</div>
<p style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-bottom:0.75rem">
  fairness_correction(b) = max(0, cat_share(b) − rec_share(b)) — ondervertegenwoordigde omroepen
  ontvangen een positieve correctie; overbelichte omroepen ontvangen 0.
  Waarden worden genormaliseerd naar [0,1] vóór menging met de CTR-score.</p>
""", unsafe_allow_html=True)

    fc_rows = []
    for b in bc_list:
        fc = fairness_correction(b, cat_share, rec_share_baseline)
        fc_rows.append({
            'Omroep':                 b,
            'Catalogusaandeel':       f"{cat_share.get(b, 0):.1%}",
            'Aanbevelingsaandeel':    f"{rec_share_baseline.get(b, 0):.1%}",
            'Eerlijkheidscorrectie':  f"{fc:.4f}",
            'Status': '✅ Ondervertegenwoordigd — versterkt' if fc > 0 else '⚠️ Overbelicht — geen versterking',
        })
    st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)
    st.divider()

    # Formulas
    st.markdown(f"<span style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:{NPO_WHITE_DIM}'>Herranking Formule</span>", unsafe_allow_html=True)
    st.latex(r"score(item) = (1 - \lambda) \times CTR\_score(item) + \lambda \times fairness\_correction_{norm}(broadcaster(item))")
    st.latex(r"fairness\_correction_{norm} = \frac{fairness\_correction(b)}{\max_{b}(fairness\_correction)}")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(
            "**Waarom normaliseren?**  \n"
            "Ruwe fairness_correction waarden zijn proportionele verschillen (~0.04–0.10). "
            "Zonder normalisatie zijn ze te klein om te concurreren met basescores (~1.0), "
            "waardoor λ geen echt effect heeft. Normaliseren naar [0,1] zorgt ervoor dat "
            "beide signalen op dezelfde schaal staan."
        )
    with col_info2:
        st.info(
            "**Mediawet 2008 vloer**  \n"
            f"λ kan niet onder 0.10 worden ingesteld. Dit handhaaft de wettelijke verplichting "
            "van NPO voor gebalanceerde omroepvertegenwoordiging. "
            "Gebruikersautonomie (Kiron Putman) werkt binnen deze vloer."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIVERSITY  (Padma Dhuney — placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_div:
    st.markdown(f"""
<div style="margin-bottom:1.5rem">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
    <div style="width:3px;height:24px;background:#3AAEA0;border-radius:2px"></div>
    <h2 style="margin:0;font-size:1.25rem">Diversity Dashboard</h2>
  </div>
  <p style="color:{NPO_WHITE_DIM};font-size:0.85rem;margin:0 0 0 13px">
    Padma Dhuney &nbsp;·&nbsp; Intra-List Similarity (ILS)
  </p>
</div>
""", unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    d1.metric("ILS — vóór herranking", f"{ils_before:.3f}",
              help="Gemiddelde paarsgewijze Jaccard-overeenkomst (hoger = minder divers)")
    d2.metric("ILS — na herranking", f"{ils_after:.3f}",
              delta=f"{ils_after - ils_before:+.3f}", delta_color="inverse")

    st.info(
        "**Dit tabblad is het onderdeel van Padma Dhuney.**  \n\n"
        "De diversity re-ranking module (`src/diversity.py`) is geïntegreerd in de pipeline "
        "en draait in Stage 2 vóór de fairness herranking. "
        "ILS-statistieken zijn zichtbaar in het tabblad Aanbevolen voor jou.  \n\n"
        "Padma bouwt hier het volledige diversity dashboard."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MIJN PROFIEL  (Kiron Putman — placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown(f"""
<div style="margin-bottom:1.5rem">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
    <div style="width:3px;height:24px;background:#8A6CC7;border-radius:2px"></div>
    <h2 style="margin:0;font-size:1.25rem">Mijn Profiel — Autonomy</h2>
  </div>
  <p style="color:{NPO_WHITE_DIM};font-size:0.85rem;margin:0 0 0 13px">
    Kiron Putman &nbsp;·&nbsp; User-controlled recommendation settings
  </p>
</div>
""", unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"<p style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:{NPO_WHITE_DIM};margin-bottom:8px'>Huidige instellingen</p>", unsafe_allow_html=True)
        st.markdown(f"**Persona:** {persona.title()}")
        st.markdown(f"**Eerlijkheidsgewicht (λ):** `{lambda_val:.2f}`")
        st.markdown(f"**Diversiteitssterkte:** `{diversity_val:.2f}`")
        pref_str = ", ".join(user_profile.get('preferred_genres', []))
        st.markdown(f"**Voorkeursgenres:** {pref_str or 'Niet ingesteld'}")
        hist = user_profile.get('watch_history', [])
        st.markdown(f"**Kijkgeschiedenisitems:** {len(hist)}")
    with p2:
        if lambda_val <= 0.15:
            st.warning("λ staat op de Mediawet 2008 minimumvloer (0.10).")
        elif lambda_val >= 0.70:
            st.success("Hoge λ — kleinere omroepen worden sterk versterkt.")
        else:
            st.info("Gebalanceerde fairness–relevantie afweging.")

    st.info(
        "**Dit tabblad is het onderdeel van Kiron Putman.**  \n\n"
        "De autonomie-besturingselementen (λ-schuifregelaar, genrevoorkeuren) zijn al "
        "gekoppeld in de zijbalk en worden doorgegeven aan de pipeline. "
        "Kiron bouwt hier de volledige profielpagina, kijkgeschiedenisweergave en voorkeurseditor."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — OVER
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(f"""
<div style="margin-bottom:1.5rem">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
    <div style="width:3px;height:24px;background:{NPO_ORANGE};border-radius:2px"></div>
    <h2 style="margin:0;font-size:1.25rem">Over dit prototype</h2>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<p style="color:{NPO_WHITE_DIM};font-size:0.9rem;margin-bottom:1.5rem">
<strong>INFOMPPM — Personalisation for (Public) Media</strong> &nbsp;·&nbsp; Utrecht University 2025–2026<br>
Een werkend aanbevelingssysteem voor NPO Start dat vier publieke waarden integreert.</p>
""", unsafe_allow_html=True)

    st.markdown("""
```
[1] Content-based scoring     — cosinus overeenkomst op genretags + populariteitsbias
        ↓
[2] Diversity re-ranking      — greedy ILS correctie  (Padma Dhuney)
        ↓
[3] Fairness re-ranking       — omroepbewuste EG correctie  (AmanDeep Singh)
                                λ-gewogen score menging · Mediawet 2008 vloer λ ≥ 0.10
        ↓
[4] Explanation labels        — leesbare reden op elke kaart  (Lisa Wang)
```
""")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Groepsleden")
        members = [
            ("AmanDeep Singh", "⚖️ Fairness", "src/fairness.py"),
            ("Padma Dhuney",   "🌍 Diversity", "src/diversity.py"),
            ("Lisa Wang",      "🔍 Transparency", "src/transparency.py"),
            ("Kiron Putman",   "🎛️ Autonomy", "src/user_profiles.py"),
        ]
        for name, value, module in members:
            st.markdown(f"""
<div style="background:{NPO_BG_CARD};border:1px solid {NPO_BG_BORDER};border-radius:6px;
            padding:0.6rem 0.8rem;margin-bottom:0.4rem;display:flex;
            align-items:center;justify-content:space-between">
  <div>
    <span style="font-weight:600;font-size:0.88rem;color:{NPO_WHITE}">{name}</span>
    <span style="font-size:0.78rem;color:{NPO_WHITE_DIM};margin-left:8px">{value}</span>
  </div>
  <code style="font-size:0.7rem;color:rgba(255,255,255,0.35)">{module}</code>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Databronnen")
        sources = [
            ("NPO Start API", "Publiek · geen authenticatie", "Observatie-baseline (rec_share)"),
            ("Synthetische catalogus", "Gegenereerd · 300 items", "Inhoudscatalogus"),
            ("Synthetische gebruikers", "Gegenereerd · 30 profielen", "6 NPO-kijkerspersona's"),
        ]
        for src, stype, use in sources:
            st.markdown(f"""
<div style="background:{NPO_BG_CARD};border:1px solid {NPO_BG_BORDER};border-radius:6px;
            padding:0.6rem 0.8rem;margin-bottom:0.4rem">
  <div style="font-weight:600;font-size:0.85rem;color:{NPO_WHITE}">{src}</div>
  <div style="font-size:0.75rem;color:{NPO_WHITE_DIM};margin-top:1px">{stype} &nbsp;·&nbsp; {use}</div>
</div>""", unsafe_allow_html=True)
