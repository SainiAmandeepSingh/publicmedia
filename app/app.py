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

# ── Broadcaster colour palette ────────────────────────────────────────────────
BROADCASTER_COLOURS = {
    'AVROTROS': '#E63946',
    'MAX':      '#F4A261',
    'KRO-NCRV': '#2A9D8F',
    'VPRO':     '#457B9D',
    'NTR':      '#6A4C93',
    'EO':       '#52B788',
    'BNNVARA':  '#E9C46A',
}

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a1a1a; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #f0f0f0 !important; }
.npo-header { background: #FF6600; padding: 1rem 1.5rem; border-radius: 8px;
              color: white; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; }
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

    users     = generate_users(cat, n_users=30, seed=42)
    cat_share = compute_cat_share(cat)
    fm, ids, _ = build_feature_matrix(cat)
    return cat, users, cat_share, rec_share_bl, fm, ids, data_source

cat, users_df, cat_share, rec_share_baseline, feature_matrix, item_ids, data_source = load_all()

# ── Pipeline: scoring → diversity re-rank → fairness re-rank ──────────────────
def run_pipeline(user_profile, lambda_weight, diversity_factor, top_n):
    # Stage 1: content-based scoring with popularity bias
    scored = score_items_for_user(cat, user_profile, feature_matrix, item_ids, DEFAULT_POPULARITY_BIAS)
    scored = apply_user_preferences(scored, user_profile)
    # Normalise base scores to [0,1] so fairness correction is proportionally meaningful
    mn, mx = scored['base_score'].min(), scored['base_score'].max()
    scored['base_score'] = (scored['base_score'] - mn) / (mx - mn + 1e-9)
    scored['current_score'] = scored['base_score']

    # Stage 2: diversity re-ranking (Padma Dhuney — ILS)
    diverse = rerank_for_diversity(scored.head(top_n * 6), top_n=top_n * 2, diversity_factor=diversity_factor)
    diverse['current_score'] = diverse['base_score']

    # Stage 3: fairness re-ranking (AmanDeep Singh — EG)
    final = rerank_for_fairness(
        diverse, cat_share, rec_share_baseline,
        lambda_weight=lambda_weight
    ).head(top_n)

    return scored, final

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📺 NPO Start")
    st.markdown("*Public Values Recommender*")
    st.divider()

    # User persona selector
    st.markdown("### 👤 Viewer Persona")
    persona = st.selectbox(
        "Select persona", list(NPO_PERSONAS.keys()),
        format_func=lambda x: x.title(),
    )
    st.caption(f"_{NPO_PERSONAS[persona]['description']}_")

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

    # ── Fairness weight — AmanDeep Singh ──────────────────────────────────────
    st.markdown("### ⚖️ Fairness Settings")
    lambda_val = st.slider(
        "Fairness weight (λ)",
        min_value=0.10, max_value=1.00,
        value=float(user_profile.get('lambda_preference', 0.5)),
        step=0.05,
        help=(
            "Controls how strongly underexposed broadcasters are boosted. "
            "λ=0.10 = near-CTR-only; λ=1.0 = maximum fairness correction. "
            "Minimum 0.10 enforced by Mediawet 2008."
        )
    )
    st.caption("⚖️ Min λ = 0.10  |  Mediawet 2008 floor")

    # ── Diversity setting — placeholder for Padma ─────────────────────────────
    st.divider()
    st.markdown("### 🌍 Diversity Settings")
    diversity_val = st.slider(
        "Diversity strength",
        min_value=0.0, max_value=1.0,
        value=float(user_profile.get('diversity_preference', 0.4)),
        step=0.05,
        help="Padma Dhuney — controls ILS reduction in diversity re-ranking."
    )

    top_n = st.slider("Recommendations to show", 6, 12, 9, step=3)

    # ── Genre preferences — placeholder for Kiron ────────────────────────────
    st.divider()
    st.markdown("### 🎭 Genre Preferences")
    all_genres = sorted(set(g for gs in cat['genres'] for g in gs))
    pref_default = [g for g in user_profile.get('preferred_genres', []) if g in all_genres]
    selected_genres = st.multiselect("Preferred genres", all_genres, default=pref_default,
                                     help="Kiron Putman — explicit genre preferences override behavioural data.")
    if selected_genres:
        user_profile['preferred_genres'] = selected_genres
        for g in selected_genres:
            user_profile.setdefault('genre_weights', {})[g] = max(
                user_profile.get('genre_weights', {}).get(g, 0.0), 0.6)

    st.divider()
    show_explanations = st.toggle("Show explanation labels", value=True,
                                  help="Lisa Wang — transparency labels on each recommendation card.")
    show_scores = st.toggle("Show score breakdown", value=False)

# ── Run the pipeline ──────────────────────────────────────────────────────────
scored_df, final_df = run_pipeline(user_profile, lambda_val, diversity_val, top_n)
baseline_top = scored_df.head(top_n).copy()
baseline_top['fairness_boosted'] = False

# ── Compute metrics ───────────────────────────────────────────────────────────
eg_before  = compute_exposure_gap(cat_share, rec_share_baseline)
eg_after   = compute_exposure_gap(cat_share, compute_rec_share(final_df))
eg_improve = (eg_before - eg_after) / eg_before * 100 if eg_before > 0 else 0
ils_before = compute_ils(baseline_top.to_dict('records'))
ils_after  = compute_ils(final_df.to_dict('records'))

# ── Page header ───────────────────────────────────────────────────────────────
st.caption(data_source)
st.markdown(
    '<div class="npo-header">📺 NPO Start — Public Values Recommender System</div>',
    unsafe_allow_html=True
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_recs, tab_fair, tab_div, tab_profile, tab_about = st.tabs([
    "🎬 Aanbevolen voor jou",
    "⚖️ Fairness",
    "🌍 Diversity",
    "👤 My Profile",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommendations
# Demonstrates the fairness intervention: side-by-side before vs after
# ══════════════════════════════════════════════════════════════════════════════
with tab_recs:
    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposure Gap — baseline", f"{eg_before:.3f}",
              help="EG before fairness re-ranking — measured from NPO Start observation data")
    c2.metric("Exposure Gap — after", f"{eg_after:.3f}",
              delta=f"{eg_after - eg_before:+.3f}", delta_color="inverse",
              help="EG after broadcaster-aware re-ranking at Stage 3")
    c3.metric("EG improvement", f"{eg_improve:.0f}%",
              help="How much the fairness gap was reduced by the re-ranking intervention")
    c4.metric("ILS reduction", f"{ils_before - ils_after:.3f}",
              help="Diversity improvement — lower ILS = more genre variety (Padma Dhuney)")
    st.divider()

    # ── Recommendation card renderer ──────────────────────────────────────────
    def render_card(item, user_profile, show_exp, show_score, col):
        b      = item.get('broadcaster', '')
        colour = BROADCASTER_COLOURS.get(b, '#999')
        boosted = item.get('fairness_boosted', False)
        border  = "border-left:4px solid #52B788;" if boosted else ""
        genres  = item.get('genres') or []
        if isinstance(genres, str):
            from src.synthetic_data import parse_genres as _pg
            genres = _pg(genres)

        genre_chips = "".join([
            f'<span style="background:{colour};color:white;padding:1px 7px;'
            f'border-radius:10px;font-size:0.68rem;margin:1px;display:inline-block">{g}</span>'
            for g in genres
        ])
        boost_badge = (
            '<span style="float:right;font-size:0.65rem;color:#52B788">⬆ Fairness boost</span>'
            if boosted else ''
        )
        col.markdown(f"""
<div style="background:#1e1e1e;border-radius:8px;padding:0.7rem;border:1px solid #333;
            {border}margin-bottom:0.3rem">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
    <span style="background:{colour};color:white;font-size:0.65rem;font-weight:700;
                 padding:1px 8px;border-radius:4px">{b}</span>{boost_badge}
  </div>
  <div style="font-size:0.88rem;font-weight:600;color:#f0f0f0;margin-bottom:4px">
    {item.get('title', '')}</div>
  <div>{genre_chips}</div>
</div>""", unsafe_allow_html=True)

        if show_exp:
            reason = get_primary_reason(item, user_profile)
            col.caption(reason)
        if show_score:
            details = get_feature_details(item, user_profile)
            with col.expander("Score breakdown"):
                for k, v in details['score_breakdown'].items():
                    st.write(f"**{k}:** {v}")

    # ── Side-by-side before / after ───────────────────────────────────────────
    col_before, col_spacer, col_after = st.columns([5, 1, 5])

    with col_before:
        st.markdown("##### 🔴 Before — CTR-only (no fairness)")
        st.caption(f"EG = {eg_before:.3f}  |  ILS = {ils_before:.3f}")
        for item in baseline_top.to_dict('records'):
            render_card(item, user_profile, show_explanations, show_scores, col_before)

    with col_spacer:
        st.markdown("")

    with col_after:
        st.markdown(f"##### 🟢 After — Fairness re-ranking (λ = {lambda_val:.2f})")
        st.caption(f"EG = {eg_after:.3f}  |  ILS = {ils_after:.3f}")
        for item in final_df.to_dict('records'):
            render_card(item, user_profile, show_explanations, show_scores, col_after)

    st.divider()

    # ── Broadcaster share in re-ranked list ───────────────────────────────────
    st.markdown("##### Broadcaster share in re-ranked list")
    bc_counts = final_df['broadcaster'].value_counts().reset_index()
    bc_counts.columns = ['Broadcaster', 'Count']
    bc_counts['Share (%)'] = bc_counts['Count'] / bc_counts['Count'].sum() * 100
    fig_bc = px.bar(
        bc_counts, x='Broadcaster', y='Share (%)',
        color='Broadcaster', color_discrete_map=BROADCASTER_COLOURS,
        text_auto='.1f', height=240,
    )
    fig_bc.update_layout(
        showlegend=False, margin=dict(t=5, b=5),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_bc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FAIRNESS DASHBOARD  (AmanDeep Singh)
# ══════════════════════════════════════════════════════════════════════════════
with tab_fair:
    st.markdown("### ⚖️ Fairness Dashboard — Producer-side Fairness")
    st.caption("**AmanDeep Singh**  |  Metric: Exposure Gap (EG)  |  Mediawet 2008")
    st.markdown(
        "EG measures how far each broadcaster's share of recommendations deviates "
        "from its share of the total catalogue. A lower EG = more equitable exposure."
    )
    st.latex(r"EG = \frac{1}{|B|} \sum_{b \in B} \left| rec\_share(b) - cat\_share(b) \right|")
    st.divider()

    # ── EG headline numbers ───────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("EG — baseline (CTR-only)", f"{eg_before:.3f}",
              help="Measured from NPO Start anonymous recommendation rows")
    colour_eg = "#2A9D8F" if eg_after < eg_before else "#E63946"
    m2.metric("EG — after re-ranking", f"{eg_after:.3f}",
              delta=f"{eg_after - eg_before:+.3f}", delta_color="inverse")
    m3.metric("EG improvement", f"{eg_improve:.0f}%",
              help="Percentage reduction in exposure gap achieved by the re-ranking intervention")
    st.divider()

    # ── Before / after bar charts ─────────────────────────────────────────────
    bc_list  = [b for b in BROADCASTER_COLOURS if b in cat_share or b in rec_share_baseline]
    cat_vals = [cat_share.get(b, 0) * 100 for b in bc_list]
    obs_vals = [rec_share_baseline.get(b, 0) * 100 for b in bc_list]
    aft_vals = [compute_rec_share(final_df).get(b, 0) * 100 for b in bc_list]

    cl, cr = st.columns(2)
    with cl:
        st.markdown(f"#### Before intervention  —  EG = **{eg_before:.3f}**")
        st.caption("Catalogue share vs observed recommendation share (CTR-optimised baseline)")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            name='Catalogue share', x=bc_list, y=cat_vals, marker_color='#457B9D'))
        fig1.add_trace(go.Bar(
            name='Observed rec share (baseline)', x=bc_list, y=obs_vals, marker_color='#E63946'))
        fig1.update_layout(
            barmode='group', height=320,
            legend=dict(orientation='h', y=-0.35),
            margin=dict(t=5, b=5), yaxis_title='Share (%)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig1, use_container_width=True)

    with cr:
        st.markdown(
            f"#### After re-ranking  —  EG = "
            f"<span style='color:{colour_eg};font-weight:700'>{eg_after:.3f}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Catalogue share vs re-ranked recommendation share (λ = {lambda_val:.2f})")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='Catalogue share', x=bc_list, y=cat_vals, marker_color='#457B9D'))
        fig2.add_trace(go.Bar(
            name='Re-ranked rec share', x=bc_list, y=aft_vals, marker_color='#52B788'))
        fig2.update_layout(
            barmode='group', height=320,
            legend=dict(orientation='h', y=-0.35),
            margin=dict(t=5, b=5), yaxis_title='Share (%)',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── λ grid search — calibration chart ────────────────────────────────────
    st.markdown("#### λ Calibration — Grid Search over Fairness Weight")
    st.caption(
        "EG is computed for every λ value in [0, 1]. "
        "The optimal λ is the lowest value that reduces EG below the target threshold "
        "while preserving acceptable engagement performance — as described in Section 3.3 of the proposal."
    )
    lambda_grid = np.arange(0.0, 1.05, 0.05)
    eg_grid = []
    for lam in lambda_grid:
        _, tmp = run_pipeline(user_profile, float(lam), diversity_val, top_n)
        eg_grid.append(compute_exposure_gap(cat_share, compute_rec_share(tmp)))

    fig_lam = go.Figure()
    fig_lam.add_trace(go.Scatter(
        x=lambda_grid, y=eg_grid,
        mode='lines+markers',
        line=dict(color='#FF6600', width=2),
        marker=dict(size=6),
        name='EG at each λ',
    ))
    fig_lam.add_vline(
        x=lambda_val, line_dash='dash', line_color='#52B788',
        annotation_text=f"Current λ = {lambda_val:.2f}",
        annotation_position="top right",
    )
    fig_lam.add_hline(
        y=eg_before, line_dash='dot', line_color='#E63946',
        annotation_text="Baseline EG (no intervention)",
    )
    fig_lam.add_hline(
        y=0.05, line_dash='dot', line_color='#F4A261',
        annotation_text="Target EG threshold = 0.05",
    )
    fig_lam.update_layout(
        height=300,
        xaxis_title='λ (fairness weight)',
        yaxis_title='Exposure Gap (EG)',
        margin=dict(t=10, b=10),
        legend=dict(orientation='h', y=-0.3),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_lam, use_container_width=True)
    st.divider()

    # ── Fairness correction per broadcaster ───────────────────────────────────
    st.markdown("#### Fairness Correction per Broadcaster")
    st.caption(
        "fairness_correction(b) = max(0, cat_share(b) − rec_share(b))  — "
        "underrepresented broadcasters receive a positive correction; "
        "overexposed broadcasters receive 0. "
        "Values are normalised to [0,1] before blending with the CTR score."
    )
    fc_rows = []
    for b in bc_list:
        fc = fairness_correction(b, cat_share, rec_share_baseline)
        fc_rows.append({
            'Broadcaster': b,
            'Catalogue share':     f"{cat_share.get(b, 0):.1%}",
            'Rec share (baseline)': f"{rec_share_baseline.get(b, 0):.1%}",
            'Fairness correction':  f"{fc:.4f}",
            'Effect': '✅ Underrepresented — receives boost' if fc > 0 else '⚠️ Overexposed — no boost',
        })
    st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)
    st.divider()

    # ── Re-ranking formula explanation ────────────────────────────────────────
    st.markdown("#### Re-ranking Formula")
    st.latex(r"score(item) = (1 - \lambda) \times CTR\_score(item) + \lambda \times fairness\_correction_{norm}(broadcaster(item))")
    st.latex(r"fairness\_correction(b) = \max(0,\ cat\_share(b) - rec\_share(b))")
    st.latex(r"fairness\_correction_{norm} = \frac{fairness\_correction}{\max(fairness\_correction)}")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(
            "**Why normalise?**  \n"
            "Raw fairness_correction values are proportional differences (~0.04–0.10). "
            "Without normalisation they are too small to compete against base scores (~1.0), "
            "making λ have no real effect. Normalising to [0,1] ensures both signals are "
            "on the same scale so λ genuinely controls the trade-off."
        )
    with col_info2:
        st.info(
            "**Mediawet 2008 floor**  \n"
            "λ cannot be set below 0.10 in this system. This enforces NPO's legal obligation "
            "under the Mediawet 2008 for balanced broadcaster representation. "
            "User autonomy (Kiron Putman) operates within this floor, not over it."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIVERSITY  (Padma Dhuney — placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_div:
    st.markdown("### 🌍 Diversity Dashboard")
    st.caption("**Padma Dhuney**  |  Metric: Intra-List Similarity (ILS)")
    st.info(
        "**This tab is Padma Dhuney's section.**  \n\n"
        "The diversity re-ranking module (`src/diversity.py`) is already integrated into the pipeline "
        "and runs at Stage 2 before the fairness re-ranking. "
        "ILS metrics are visible in Tab 1 (Recommendations).  \n\n"
        "Padma will build out the full diversity dashboard here."
    )
    # Minimal metrics so the tab isn't empty
    d1, d2 = st.columns(2)
    d1.metric("ILS — before re-ranking", f"{ils_before:.3f}",
              help="Mean pairwise Jaccard similarity across recommended items (higher = less diverse)")
    d2.metric("ILS — after re-ranking", f"{ils_after:.3f}",
              delta=f"{ils_after - ils_before:+.3f}", delta_color="inverse")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MY PROFILE / AUTONOMY  (Kiron Putman — placeholder)
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown("### 👤 My Profile — Autonomy")
    st.caption("**Kiron Putman**  |  User-controlled recommendation settings")
    st.info(
        "**This tab is Kiron Putman's section.**  \n\n"
        "The autonomy controls (λ slider, genre preferences) are already wired into the sidebar "
        "and feed into the pipeline. "
        "Kiron will build out the full profile page, watch history view, and preference editor here."
    )
    # Show current settings so the tab has something useful
    st.markdown("#### Current settings (from sidebar)")
    st.markdown(f"**Persona:** {persona.title()}")
    st.markdown(f"**Fairness weight (λ):** `{lambda_val:.2f}`")
    st.markdown(f"**Diversity strength:** `{diversity_val:.2f}`")
    pref_str = ", ".join(user_profile.get('preferred_genres', []))
    st.markdown(f"**Preferred genres:** {pref_str or 'None set'}")
    hist = user_profile.get('watch_history', [])
    st.markdown(f"**Watch history items:** {len(hist)}")
    if lambda_val <= 0.15:
        st.warning("λ is at the Mediawet 2008 minimum floor (0.10).")
    elif lambda_val >= 0.70:
        st.success("High λ — smaller broadcasters are strongly boosted.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About this prototype")
    st.markdown("""
**INFOMPPM — Personalisation for (Public) Media** | Utrecht University 2025–2026

This prototype integrates four public values into a recommender system for **NPO Start**,
the on-demand platform of the Nederlandse Publieke Omroep.

#### Pipeline

```
[1] Content-based scoring     — cosine similarity on genre tags
                                + popularity bias (CTR-driven baseline)
        ↓
[2] Diversity re-ranking      — greedy ILS correction (Padma Dhuney)
        ↓
[3] Fairness re-ranking       — broadcaster-aware EG correction (AmanDeep Singh)
                                λ-weighted score blending, Mediawet 2008 floor λ ≥ 0.10
        ↓
[4] Explanation labels        — human-readable reason on each card (Lisa Wang)
```

#### Group members

| Name | Public Value | Module |
|---|---|---|
| AmanDeep Singh | **Fairness** — equitable broadcaster exposure | `src/fairness.py` |
| Padma Dhuney | **Diversity** — content variety for users | `src/diversity.py` |
| Lisa Wang | **Transparency** — explanations in the interface | `src/transparency.py` |
| Kiron Putman | **Autonomy** — user control over recommendations | `src/user_profiles.py` |

#### Data sources

| Source | Type | Used for |
|---|---|---|
| NPO Start `npo.nl/start/api` | Real — public, no auth | Observation baseline (`rec_share`) |
| Synthetic catalogue | Generated — 300 items | Content catalogue (broadcaster, genre, metadata) |
| Synthetic users | Generated — 30 profiles | 6 NPO viewer personas |
    """)
