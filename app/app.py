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
from src.transparency import get_primary_reason, get_feature_details, get_algorithm_explainer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPO Start · Public Values Recommender",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── NPO Design tokens (extracted from npo.nl/start computed styles) ───────────
NPO_BG_DEEP   = "#0A1931"
NPO_BG_CARD   = "#081426"
NPO_BG_MID    = "#1F3353"
NPO_BG_BORDER = "#293D5D"
NPO_ORANGE    = "#F56A00"
NPO_WHITE     = "#FFFFFF"
NPO_WHITE_DIM = "rgba(255,255,255,0.60)"
NPO_WHITE_SUB = "rgba(255,255,255,0.35)"

BROADCASTER_COLOURS = {
    'AVROTROS': '#E05252',
    'MAX':      '#E8953A',
    'KRO-NCRV': '#3AAEA0',
    'VPRO':     '#5B8FCC',
    'NTR':      '#8A6CC7',
    'EO':       '#5BBF8A',
    'BNNVARA':  '#D4A843',
    'PowNed':   '#A0A0A0',
    'WNL':      '#8B7355',
    'HUMAN':    '#5E9E6E',
    'NPO Zapp': '#E07B39',
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"], .main {{
  background-color: {NPO_BG_DEEP} !important;
  color: {NPO_WHITE} !important;
  font-family: 'Inter', 'NPO Sans', sans-serif !important;
}}
[data-testid="stMainBlockContainer"],
[data-testid="block-container"] {{
  background-color: {NPO_BG_DEEP} !important;
  padding-top: 0.5rem !important;
}}
[data-testid="stSidebar"] {{
  background-color: {NPO_BG_CARD} !important;
  border-right: 1px solid {NPO_BG_BORDER} !important;
}}
[data-testid="stSidebar"] * {{ color: {NPO_WHITE} !important; }}

/* Tabs */
button[data-baseweb="tab"] {{
  background: transparent !important;
  color: {NPO_WHITE_DIM} !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  padding: 0.6rem 1.2rem !important;
  border-bottom: 3px solid transparent !important;
  border-radius: 0 !important;
}}
button[data-baseweb="tab"]:hover {{ color: {NPO_WHITE} !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: {NPO_WHITE} !important;
  border-bottom: 3px solid {NPO_ORANGE} !important;
}}
[data-testid="stTabs"] > div:first-child {{
  border-bottom: 1px solid {NPO_BG_BORDER} !important;
}}
[data-testid="stTabsContent"] {{
  background: transparent !important;
  padding-top: 1.25rem !important;
}}

/* Metrics */
[data-testid="metric-container"] {{
  background: {NPO_BG_CARD} !important;
  border: 1px solid {NPO_BG_BORDER} !important;
  border-radius: 8px !important;
  padding: 1rem 1.25rem !important;
}}
[data-testid="metric-container"] label {{
  color: {NPO_WHITE_DIM} !important;
  font-size: 0.72rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
  color: {NPO_WHITE} !important;
  font-size: 1.8rem !important;
  font-weight: 700 !important;
}}

/* Info boxes */
[data-testid="stInfo"] {{
  background: rgba(91,143,204,0.10) !important;
  border-left: 3px solid #5B8FCC !important;
  border-radius: 0 6px 6px 0 !important;
  color: {NPO_WHITE} !important;
}}
[data-testid="stWarning"] {{
  background: rgba(245,106,0,0.10) !important;
  border-left: 3px solid {NPO_ORANGE} !important;
  color: {NPO_WHITE} !important;
}}
[data-testid="stSuccess"] {{
  background: rgba(91,191,138,0.10) !important;
  border-left: 3px solid #5BBF8A !important;
  color: {NPO_WHITE} !important;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
  background: {NPO_BG_CARD} !important;
  border-radius: 8px !important;
  border: 1px solid {NPO_BG_BORDER} !important;
}}

/* Inputs */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div {{
  background: {NPO_BG_MID} !important;
  border: 1px solid {NPO_BG_BORDER} !important;
  border-radius: 6px !important;
}}

/* Typography */
h1, h2, h3 {{
  color: {NPO_WHITE} !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {NPO_BG_DEEP}; }}
::-webkit-scrollbar-thumb {{ background: {NPO_BG_BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {NPO_ORANGE}; }}

/* Remove all Streamlit default dividers */
hr {{ border-color: {NPO_BG_BORDER} !important; opacity: 0.5 !important; }}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Real NPO broadcaster catalogue share — derived from NPO Start broadcaster pages
# via the page-layout + page-collection API (fetched by data_loader.py).
# Falls back to hardcoded estimates if cat_share.json is not available.
# Source: live NPO Start API (npo.nl/start/api/domain/page-layout?layoutId=<broadcaster>)
# Fetched values (719 items across 7 broadcasters):
#   EO: 29.9%, VPRO: 18.5%, KRO-NCRV: 17.9%, BNNVARA: 14.9%,
#   NTR: 8.9%, MAX: 5.3%, AVROTROS: 4.6%
REAL_CAT_SHARE_FALLBACK = {
    "AVROTROS": 0.0459,
    "BNNVARA":  0.1488,
    "EO":       0.2990,
    "KRO-NCRV": 0.1794,
    "MAX":       0.0529,
    "NTR":       0.0890,
    "VPRO":      0.1850,
}

@st.cache_data
def load_all():
    cat_path = DATA_DIR / "catalogue.csv"
    rs_path  = DATA_DIR / "rec_share.json"

    if cat_path.exists():
        cat = pd.read_csv(cat_path)
        cat['genres'] = cat['genres'].apply(parse_genres)
        if 'item_id' not in cat.columns and 'slug' in cat.columns:
            cat = cat.rename(columns={'slug': 'item_id'})
        if 'image_url' not in cat.columns:
            cat['image_url'] = ""
        # Load rec_share from real NPO Start observation data
        rec_share_bl = json.loads(rs_path.read_text()) if rs_path.exists() \
                       else compute_rec_share(cat)
        # Load real catalogue share from data_loader.py output if available.
        # This uses empirically-derived proportions from NPO Start's broadcaster
        # pages (page-layout + page-collection API), giving the most accurate
        # cat_share for the Exposure Gap computation.
        cs_path = DATA_DIR / "cat_share.json"
        if cs_path.exists():
            cat_share = json.loads(cs_path.read_text())
        else:
            # Fallback to hardcoded estimates from NPO Start API (March 2026)
            cat_share = REAL_CAT_SHARE_FALLBACK
        cat_share_note = " · real catalogue" if cs_path.exists() else " · estimated catalogue"
        data_source = f"🟢 Real data · {len(cat)} NPO series{cat_share_note}"
    else:
        cat = generate_catalogue(n_items=300, seed=42)
        obs = generate_observation_sample(cat, n_sessions=200, seed=42)
        rec_share_bl = compute_rec_share(obs)
        cat['image_url'] = ""
        # For synthetic data, compute cat_share from the generated catalogue
        cat_share = compute_cat_share(cat)
        data_source = f"🟡 Synthetic data · {len(cat)} items"

    users      = generate_users(cat, n_users=30, seed=42)
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
    diverse = rerank_for_diversity(
        scored.head(top_n * 6), top_n=top_n * 2, diversity_factor=diversity_factor)
    final = rerank_for_fairness(
        diverse, cat_share, rec_share_baseline, lambda_weight=lambda_weight
    ).head(top_n)
    return scored, final

# ── Chart theme ───────────────────────────────────────────────────────────────
def npo_layout(**kw):
    base = dict(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=NPO_WHITE, family='Inter, sans-serif', size=12),
        xaxis=dict(gridcolor=NPO_BG_BORDER, linecolor=NPO_BG_BORDER,
                   tickfont=dict(color=NPO_WHITE_DIM), title_font=dict(color=NPO_WHITE_DIM)),
        yaxis=dict(gridcolor=NPO_BG_BORDER, linecolor=NPO_BG_BORDER,
                   tickfont=dict(color=NPO_WHITE_DIM), title_font=dict(color=NPO_WHITE_DIM)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=NPO_WHITE_DIM),
                    orientation='h', y=-0.3),
        margin=dict(t=10, b=10, l=5, r=5),
    )
    base.update(kw)
    return base

# ── Helper: section header ────────────────────────────────────────────────────
def section_header(title, subtitle, accent_colour=None):
    colour = accent_colour or NPO_ORANGE
    st.markdown(f"""
<div style="margin-bottom:1.25rem">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:3px">
    <div style="width:3px;height:22px;background:{colour};border-radius:2px;flex-shrink:0"></div>
    <h2 style="margin:0;font-size:1.2rem;font-weight:700">{title}</h2>
  </div>
  <p style="color:{NPO_WHITE_DIM};font-size:0.82rem;margin:0 0 0 13px">{subtitle}</p>
</div>
""", unsafe_allow_html=True)

def label(text):
    st.markdown(
        f"<p style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.08em;"
        f"color:{NPO_WHITE_SUB};margin-bottom:4px;margin-top:0'>{text}</p>",
        unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
<div style="padding:1rem 0 0.75rem 0;display:flex;align-items:center;gap:10px">
  <div style="background:{NPO_ORANGE};width:34px;height:34px;border-radius:6px;
              display:flex;align-items:center;justify-content:center;
              font-weight:900;font-size:12px;color:white;flex-shrink:0">NPO</div>
  <div>
    <div style="font-weight:700;font-size:0.92rem">NPO Start</div>
    <div style="font-size:0.7rem;color:rgba(255,255,255,0.4)">Public Values Prototype</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.divider()

    label("Viewer Persona")
    persona = st.selectbox("persona", list(NPO_PERSONAS.keys()),
                           format_func=lambda x: x.title(),
                           label_visibility="collapsed")
    st.markdown(f"<p style='font-size:0.76rem;color:rgba(255,255,255,0.45);margin-top:-4px;font-style:italic'>{NPO_PERSONAS[persona]['description']}</p>", unsafe_allow_html=True)

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

    label("⚖️ Fairness Weight (λ)")
    lambda_val = st.slider("lam", 0.10, 1.00,
                           float(user_profile.get('lambda_preference', 0.5)), 0.05,
                           label_visibility="collapsed",
                           help="Controls how strongly underexposed broadcasters are boosted. Min 0.10 = Mediawet 2008 floor.")
    st.markdown(f"<p style='font-size:0.7rem;color:rgba(255,255,255,0.35);margin-top:-2px'>Min 0.10  ·  Mediawet 2008 floor</p>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:0.75rem;color:rgba(255,255,255,0.50);margin-top:4px;line-height:1.4'>"
        f"λ = 0.10 replicates the current CTR-only system. "
        f"λ = 1.0 applies maximum fairness correction. "
        f"Values in between balance relevance and equitable broadcaster exposure.</p>",
        unsafe_allow_html=True
    )

    st.divider()
    label("🌍 Diversity Strength")
    diversity_val = st.slider("div", 0.0, 1.0,
                              float(user_profile.get('diversity_preference', 0.4)), 0.05,
                              label_visibility="collapsed",
                              help="Controls genre variety in recommendations via ILS re-ranking.")
    st.markdown(
        f"<p style='font-size:0.75rem;color:rgba(255,255,255,0.50);margin-top:4px;line-height:1.4'>"
        f"0.0 = no diversity correction, pure relevance ranking. "
        f"1.0 = maximum genre variety. "
        f"Values in between balance relevance and content diversity.</p>",
        unsafe_allow_html=True
    )

    st.divider()
    label("📺 Recommendations")
    top_n = st.slider("top_n", 4, 12, 8, step=1,
                      label_visibility="collapsed",
                      help="Number of programmes to show in each list.")

    st.divider()
    label("🎭 Genre Preferences")
    all_genres   = sorted(set(g for gs in cat['genres'] for g in gs))
    pref_default = [g for g in user_profile.get('preferred_genres', []) if g in all_genres]
    selected_genres = st.multiselect("genres", all_genres, default=pref_default,
                                     label_visibility="collapsed",
                                     help="Explicit genre preferences that override behavioural data.")
    if selected_genres:
        user_profile['preferred_genres'] = selected_genres
        for g in selected_genres:
            user_profile.setdefault('genre_weights', {})[g] = max(
                user_profile.get('genre_weights', {}).get(g, 0.0), 0.6)

    st.divider()
    show_explanations = st.toggle("Show explanation labels", value=True)
    show_scores       = st.toggle("Show full explanation",    value=False)

    st.divider()
    from pathlib import Path as _P
    _cat_exists = (_P(__file__).parent.parent / "data" / "processed" / "catalogue.csv").exists()
    if _cat_exists:
        st.markdown(
            f"<p style='font-size:0.7rem;color:#5BBF8A;line-height:1.4'>"
            f"🟢 Real NPO data loaded</p>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<p style='font-size:0.7rem;color:rgba(255,255,255,0.35);line-height:1.4'>"
            f"🟡 Using synthetic data<br>"
            f"Run <code>python src/data_loader.py</code> for real NPO content</p>",
            unsafe_allow_html=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
scored_df, final_df = run_pipeline(user_profile, lambda_val, diversity_val, top_n)
baseline_top = scored_df.head(top_n).copy()
baseline_top['fairness_boosted'] = False

eg_before  = compute_exposure_gap(cat_share, rec_share_baseline)
eg_after   = compute_exposure_gap(cat_share, compute_rec_share(final_df))
eg_improve = (eg_before - eg_after) / eg_before * 100 if eg_before > 0.001 else None
ils_before = compute_ils(baseline_top.to_dict('records'))
ils_after  = compute_ils(final_df.to_dict('records'))

# ── Top navigation bar ────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:{NPO_BG_CARD};border-bottom:1px solid {NPO_BG_BORDER};
            padding:0.65rem 1.5rem;margin:-1rem -1rem 1.25rem -1rem;
            display:flex;align-items:center;justify-content:space-between">
  <div style="display:flex;align-items:center;gap:10px">
    <div style="background:{NPO_ORANGE};padding:3px 9px;border-radius:5px;
                font-weight:900;font-size:13px;color:white">NPO</div>
    <span style="font-weight:600;font-size:1rem">Start</span>
    <span style="color:{NPO_WHITE_SUB};font-size:0.8rem">·</span>
    <span style="color:{NPO_WHITE_SUB};font-size:0.8rem">Public Values Recommender</span>
  </div>
  <span style="font-size:0.72rem;color:{NPO_WHITE_SUB}">{data_source}</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_recs, tab_fair, tab_about = st.tabs([
    "Recommended for You",
    "⚖️ Fairness",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommended for You
# ══════════════════════════════════════════════════════════════════════════════
with tab_recs:
    with st.expander("❓ How does the algorithm work?"):
        st.markdown(get_algorithm_explainer())

    # NPO-style card: image fills card, broadcaster badge top-left, 2 info lines below
    def render_card_npo(item, user_profile, show_exp, show_score, col):
        b       = item.get("broadcaster", "")
        colour  = BROADCASTER_COLOURS.get(b, "#6B7A99")
        boosted = item.get("fairness_boosted", False)
        genres  = item.get("genres") or []
        if isinstance(genres, str):
            genres = parse_genres(genres)
        title   = item.get("title", "").replace("<", "&lt;").replace(">", "&gt;")
        img_url = (item.get("image_url") or "").strip()
        genre_text = "  ·  ".join(genres[:2]) if genres else ""

        # Fairness boost badge — top-right
        boost = (
            "<div style='position:absolute;top:8px;right:8px;"
            "background:#5BBF8A;color:white;font-size:0.6rem;font-weight:700;"
            "padding:2px 8px;border-radius:3px;z-index:2'>&#x2B06; boost</div>"
        ) if boosted else ""

        # Broadcaster badge — top-left, coloured
        bc_badge = (
            "<div style='position:absolute;top:8px;left:8px;"
            "background:" + colour + ";color:white;"
            "font-size:0.6rem;font-weight:700;padding:2px 8px;border-radius:4px;"
            "letter-spacing:0.04em;z-index:2'>"
            + b + "</div>"
        )

        # Title gradient overlay at bottom of image
        title_overlay = (
            "<div style='position:absolute;bottom:0;left:0;right:0;height:55%;"
            "background:linear-gradient(transparent," + NPO_BG_DEEP + "EE)'></div>"
            "<div style='position:absolute;bottom:8px;left:10px;right:10px;"
            "font-size:0.87rem;font-weight:700;color:#FFFFFF;line-height:1.2;"
            "text-shadow:0 1px 4px rgba(0,0,0,0.9)'>"
            + title + "</div>"
        )

        if img_url:
            card = (
                "<div style='position:relative;border-radius:10px;"
                "overflow:hidden;margin-bottom:4px'>"
                "<img src='" + img_url + "' "
                "style='width:100%;aspect-ratio:16/9;object-fit:cover;display:block'>"
                + bc_badge + boost + title_overlay +
                "</div>"
            )
        else:
            card = (
                "<div style='position:relative;border-radius:10px;"
                "overflow:hidden;margin-bottom:4px;"
                "background:" + NPO_BG_MID + ";aspect-ratio:16/9'>"
                + bc_badge + boost + title_overlay +
                "</div>"
            )

        col.markdown(card, unsafe_allow_html=True)

        # Line 1: genre — always shown
        if genre_text:
            col.markdown(
                "<p style='margin:2px 0 0 2px;font-size:0.72rem;"
                "color:rgba(255,255,255,0.50);white-space:nowrap;"
                "overflow:hidden;text-overflow:ellipsis'>"
                + genre_text + "</p>",
                unsafe_allow_html=True)

        # Line 2: explanation reason — always shown when show_exp
        if show_exp:
            reason = get_primary_reason(item, user_profile)
            # Strip the emoji and clean up
            r = reason[2:].strip() if reason and len(reason) > 2 else reason
            col.markdown(
                "<p style='margin:2px 0 10px 2px;font-size:0.68rem;"
                "color:rgba(255,255,255,0.30);white-space:nowrap;"
                "overflow:hidden;text-overflow:ellipsis'>"
                + r + "</p>",
                unsafe_allow_html=True)
        else:
            # Always maintain spacing below card even without explanation
            col.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

        if show_score:
            details = get_feature_details(item, user_profile)
            icon_map = {
                "Matches": "🎯",
                "Boosted": "⚖️",
                "Selected": "🌍",
                "Recently": "🆕",
                "Popular": "🔥"
            }            
            with col.expander("ℹ️ Why this recommendation?"):
                for f in details["features"]:
                    icon = "•"
                    for key in icon_map:
                        if f.startswith(key):
                            icon = icon_map[key]
                    st.write(f"{icon} {f}")

                with st.expander("See technical details"):
                    st.write("**Score breakdown**")
                    for k, v in details["score_breakdown"].items():
                        st.caption(f"{k}: {v:.3f}")

    # 2-column grid per side (wider cards, closer to real NPO)
    def render_grid(items, user_profile, show_exp, show_score, suppress_score=False):
        for row_start in range(0, len(items), 2):
            row = items[row_start:row_start + 2]
            cols = st.columns(2)
            for ci, item in enumerate(row):
                render_card_npo(item, user_profile, show_exp,
                                False if suppress_score else show_score,
                                cols[ci])

    # Section headers
    hl, hr = st.columns(2)
    with hl:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            "<div style='width:9px;height:9px;background:#E05252;border-radius:50%'></div>"
            "<span style='font-size:0.9rem;font-weight:700'>CTR Only  ·  No Fairness Correction</span>"
            "</div>"
            f"<p style='font-size:0.73rem;color:{NPO_WHITE_DIM};margin:0 0 8px 17px'>"
            f"EG {eg_before:.3f}  ·  ILS {ils_before:.3f}</p>",
            unsafe_allow_html=True)
    with hr:
        st.markdown(
            "<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            "<div style='width:9px;height:9px;background:#5BBF8A;border-radius:50%'></div>"
            f"<span style='font-size:0.9rem;font-weight:700'>After Re-ranking  ·  λ = {lambda_val:.2f}</span>"
            "</div>"
            f"<p style='font-size:0.73rem;color:{NPO_WHITE_DIM};margin:0 0 8px 17px'>"
            f"EG {eg_after:.3f}  ·  ILS {ils_after:.3f}</p>",
            unsafe_allow_html=True)

    # Before / After grids
    gl, gm, gr = st.columns([9, 1, 9])
    with gl:
        render_grid(baseline_top.to_dict("records"), user_profile,
                    show_explanations, show_scores, suppress_score=True)
    with gm:
        st.empty()
    with gr:
        render_grid(final_df.to_dict("records"), user_profile,
                    show_explanations, show_scores)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FAIRNESS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_fair:
    section_header(
        "Fairness Dashboard  ·  Producer-side Fairness",
        "Exposure Gap (EG) metric  ·  Mediawet 2008  ·  Proportional fairness  ·  Stage 3 re-ranking",
        NPO_ORANGE
    )
    st.markdown(
        f"<p style='color:{NPO_WHITE_DIM};font-size:0.88rem;margin-bottom:0.75rem'>"
        "EG measures how far each broadcaster's recommendation share deviates from "
        "its catalogue share. Lower EG means more equitable exposure.</p>",
        unsafe_allow_html=True)
    st.latex(r"EG = \frac{1}{|B|} \sum_{b \in B} \left| rec\_share(b) - cat\_share(b) \right|")
    # st.divider()

    # KPIs
    m1, m2, m3 = st.columns(3)
    m1.metric("EG Baseline (CTR only)", f"{eg_before:.3f}")
    m2.metric("EG After Re-ranking",    f"{eg_after:.3f}",
              delta=f"{eg_after - eg_before:+.3f}", delta_color="inverse")
    m3.metric("EG Improvement", f"{eg_improve:.0f}%" if eg_improve is not None else "N/A")

    # Contextual explanation when EG worsens
    if eg_after > eg_before:
        st.warning(
            "**Why did EG increase?**  \n"
            "The fairness correction boosts underrepresented broadcasters (EO, VPRO) whose "
            "content matches this persona's preferences. This causes over-concentration in "
            "those broadcasters rather than spreading exposure. This is the "
            "**fairness-relevance tension** from Section 4 of the proposal. "
            "Try the **Varied Consumer** or **Family Viewer** persona to see EG reduction."
        )
    elif eg_improve is not None and eg_improve > 20:
        st.success(
            f"**EG reduced by {eg_improve:.0f}%.** "
            "The re-ranking redistributed exposure toward underrepresented broadcasters "
            "while preserving relevance for this viewer profile."
        )
    st.divider()

    # Before / after charts
    bc_list  = [b for b in BROADCASTER_COLOURS if b in cat_share or b in rec_share_baseline]
    cat_vals = [round(cat_share.get(b, 0) * 100, 1) for b in bc_list]
    obs_vals = [round(rec_share_baseline.get(b, 0) * 100, 1) for b in bc_list]
    aft_vals = [round(compute_rec_share(final_df).get(b, 0) * 100, 1) for b in bc_list]

    cl, cr = st.columns(2)
    with cl:
        label("Before intervention")
        st.markdown(
            f'<span style="background:#E05252;color:white;font-size:0.72rem;font-weight:700;'
            f'padding:2px 9px;border-radius:4px">EG = {eg_before:.3f}</span>',
            unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name='Catalogue share', x=bc_list, y=cat_vals,
                              marker_color='#5B8FCC', marker_line_width=0))
        fig1.add_trace(go.Bar(name='Observed rec share (baseline)', x=bc_list, y=obs_vals,
                              marker_color='#E05252', marker_line_width=0))
        fig1.update_layout(**npo_layout(barmode='group', height=290,
                                        yaxis_title='Share (%)'))
        st.plotly_chart(fig1, use_container_width=True)

    with cr:
        label(f"After re-ranking (λ = {lambda_val:.2f})")
        colour_eg = "#5BBF8A" if eg_after < eg_before else "#E05252"
        st.markdown(
            f'<span style="background:{colour_eg};color:white;font-size:0.72rem;font-weight:700;'
            f'padding:2px 9px;border-radius:4px">EG = {eg_after:.3f}</span>',
            unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Catalogue share', x=bc_list, y=cat_vals,
                              marker_color='#5B8FCC', marker_line_width=0))
        fig2.add_trace(go.Bar(name='Re-ranked rec share', x=bc_list, y=aft_vals,
                              marker_color='#5BBF8A', marker_line_width=0))
        fig2.update_layout(**npo_layout(barmode='group', height=290,
                                        yaxis_title='Share (%)'))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Lambda calibration chart
    label("λ Calibration · Grid Search over Fairness Weight")
    st.markdown(
        f"<p style='font-size:0.8rem;color:rgba(255,255,255,0.45);margin-bottom:0.75rem'>"
        "EG is computed for every λ value in [0,1]. The optimal λ is the lowest value that "
        "reduces EG below the target threshold while preserving engagement performance "
        "(Section 3.3 of the proposal).</p>",
        unsafe_allow_html=True)

    with st.spinner("Computing λ calibration across all fairness weights..."):
        lambda_grid = np.arange(0.0, 1.05, 0.05)
        eg_grid = []
        for lam in lambda_grid:
            _, tmp = run_pipeline(user_profile, float(lam), diversity_val, top_n)
            eg_grid.append(round(compute_exposure_gap(cat_share, compute_rec_share(tmp)), 4))

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
    fig_lam.add_hline(y=0.05, line_dash='dot', line_color=NPO_WHITE_SUB, line_width=1,
                      annotation_text="Target threshold EG = 0.05",
                      annotation_font=dict(color=NPO_WHITE_SUB, size=10))
    fig_lam.update_layout(**npo_layout(
        height=290, xaxis_title='λ (fairness weight)', yaxis_title='Exposure Gap (EG)'))
    st.plotly_chart(fig_lam, use_container_width=True)

    # Note when threshold not reached
    min_eg = min(eg_grid)
    if min_eg > 0.05:
        opt_lam = lambda_grid[eg_grid.index(min_eg)]
        st.info(
            f"**EG target threshold (0.05) not reached for this persona.** "
            f"Minimum EG achieved: **{min_eg:.3f}** at λ = {opt_lam:.2f}. "
            "When user preferences already align with underrepresented broadcasters, "
            "the correction amplifies their exposure rather than distributing it. "
            "Try **Varied Consumer** to see the threshold crossed."
        )
    else:
        crossed = [(lam, eg) for lam, eg in zip(lambda_grid, eg_grid) if eg <= 0.05]
        if crossed:
            opt_lam, opt_eg = crossed[0]
            st.success(
                f"**Optimal λ = {opt_lam:.2f}** reduces EG to **{opt_eg:.3f}**, "
                "below the 0.05 target. This is the recommended deployment value "
                "under the Mediawet 2008 mandate."
            )
    st.divider()

    # Fairness correction table
    label("Fairness Correction per Broadcaster")
    st.markdown(
        f"<p style='font-size:0.8rem;color:rgba(255,255,255,0.45);margin-bottom:0.75rem'>"
        "fairness_correction(b) = max(0, cat_share(b) − rec_share(b))  · "
        "underrepresented broadcasters receive a positive correction. "
        "Values are normalised to [0,1] before blending with the CTR score.</p>",
        unsafe_allow_html=True)

    fc_rows = []
    for b in bc_list:
        fc = fairness_correction(b, cat_share, rec_share_baseline)
        fc_rows.append({
            'Broadcaster':          b,
            'Catalogue Share':      f"{cat_share.get(b, 0):.1%}",
            'Rec Share (baseline)': f"{rec_share_baseline.get(b, 0):.1%}",
            'Fairness Correction':  f"{fc:.4f}",
            'Status': '✅ Underrepresented' if fc > 0 else '⚠️ Overexposed',
        })
    st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)
    st.divider()

    # Formulas
    label("Re-ranking Formula")
    st.latex(r"score(item) = (1 - \lambda) \times CTR\_score(item) + \lambda \times fairness\_correction_{norm}(broadcaster(item))")
    st.latex(r"fairness\_correction_{norm}(b) = \frac{\max(0,\ cat\_share(b) - rec\_share(b))}{\max_b(\max(0,\ cat\_share(b) - rec\_share(b)))}")

    i1, i2 = st.columns(2)
    with i1:
        st.info(
            "**Why normalise?**  \n"
            "Raw fairness correction values are small (~0.04 to 0.10). "
            "Without normalisation they cannot compete against base scores (~1.0) "
            "and λ has no real effect. Normalising to [0,1] puts both signals "
            "on the same scale so λ genuinely controls the trade-off."
        )
    with i2:
        st.info(
            "**Mediawet 2008 floor**  \n"
            "λ cannot be set below 0.10. This enforces NPO's legal obligation "
            "for balanced broadcaster representation under Dutch media law. "
            "User autonomy (Kiron Putman) operates within this floor, not above it."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    section_header("About This Prototype", "INFOMPPM  ·  Utrecht University 2025-2026")

    st.markdown(f"""
<p style="color:{NPO_WHITE_DIM};font-size:0.9rem;max-width:760px;margin-bottom:0.75rem;line-height:1.6">
A working recommender system prototype for <strong style="color:{NPO_WHITE}">NPO Start</strong>,
the on-demand platform of the Nederlandse Publieke Omroep. The system addresses a structural
fairness gap in NPO's CTR-optimised recommendation pipeline, which systematically underexposes
smaller member broadcasters (VPRO, NTR, EO) in violation of the
<strong style="color:{NPO_WHITE}">Mediawet 2008</strong> mandate for balanced representation.
</p>
<p style="color:{NPO_WHITE_DIM};font-size:0.9rem;max-width:760px;margin-bottom:1.5rem;line-height:1.6">
The prototype integrates four public values (fairness, diversity, transparency, and autonomy)
into a single pipeline, demonstrating how algorithmic design choices can be made to reflect
institutional obligations rather than pure engagement metrics.
</p>
""", unsafe_allow_html=True)

    st.markdown("""
```
[1] Content-based scoring    cosine similarity on genre tags + popularity bias
        ↓
[2] Diversity re-ranking     greedy ILS correction  (Padma Dhuney)
        ↓
[3] Fairness re-ranking     broadcaster-aware EG correction  (AmanDeep Singh)
                               λ-weighted score blending  ·  Mediawet 2008 floor λ ≥ 0.10
        ↓
[4] Explanation labels      human-readable reason on each card  (Lisa Wang)
```
""")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        label("Group Members")
        members = [
            ("AmanDeep Singh", "⚖️ Fairness",      "src/fairness.py",      NPO_ORANGE),
            ("Padma Dhuney",   "🌍 Diversity",     "src/diversity.py",     "#3AAEA0"),
            ("Lisa Wang",      "🔍 Transparency",  "src/transparency.py",  "#5B8FCC"),
            ("Kiron Putman",   "🎛️ Autonomy",      "src/user_profiles.py", "#8A6CC7"),
        ]
        for name, value, module, colour in members:
            st.markdown(f"""
<div style="background:{NPO_BG_CARD};border:1px solid {NPO_BG_BORDER};
            border-left:3px solid {colour};border-radius:6px;
            padding:0.55rem 0.8rem;margin-bottom:0.4rem;
            display:flex;align-items:center;justify-content:space-between">
  <div>
    <span style="font-weight:600;font-size:0.87rem">{name}</span>
    <span style="font-size:0.78rem;color:{NPO_WHITE_DIM};margin-left:8px">{value}</span>
  </div>
  <code style="font-size:0.68rem;color:rgba(255,255,255,0.3)">{module}</code>
</div>""", unsafe_allow_html=True)

    with c2:
        label("Data Sources")
        sources = [
            ("NPO Start API",       "Public · no authentication", "Observation baseline (rec_share)", NPO_ORANGE),
            ("Synthetic Catalogue", "Generated · 300 items",      "Content catalogue",                "#5B8FCC"),
            ("Synthetic Users",     "Generated · 30 profiles",    "6 NPO viewer personas",            "#3AAEA0"),
        ]
        for src, stype, use, colour in sources:
            st.markdown(f"""
<div style="background:{NPO_BG_CARD};border:1px solid {NPO_BG_BORDER};
            border-left:3px solid {colour};border-radius:6px;
            padding:0.55rem 0.8rem;margin-bottom:0.4rem">
  <div style="font-weight:600;font-size:0.85rem">{src}</div>
  <div style="font-size:0.75rem;color:{NPO_WHITE_DIM};margin-top:2px">{stype}  ·  {use}</div>
</div>""", unsafe_allow_html=True)
