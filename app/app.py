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

from src.synthetic_data import generate_catalogue, generate_observation_sample
from src.user_profiles import generate_users, apply_user_preferences, NPO_PERSONAS
from src.scoring import build_feature_matrix, score_items_for_user, DEFAULT_POPULARITY_BIAS
from src.diversity import rerank_for_diversity, compute_ils
from src.fairness import (
    compute_cat_share, compute_rec_share, compute_exposure_gap,
    rerank_for_fairness, fairness_correction,
)
from src.transparency import get_primary_reason, get_feature_details, get_algorithm_explainer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPO Start — Public Values Recommender",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BROADCASTER_COLOURS = {
    'AVROTROS': '#E63946', 'MAX': '#F4A261', 'KRO-NCRV': '#2A9D8F',
    'VPRO': '#457B9D', 'NTR': '#6A4C93', 'EO': '#52B788',
}

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1a1a1a; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #f0f0f0 !important; }
.npo-header { background: #FF6600; padding: 1rem 1.5rem; border-radius: 8px;
              color: white; font-size: 1.3rem; font-weight: 700; margin-bottom: 1rem; }
.card-dark { background:#1e1e1e; border-radius:10px; padding:0.8rem;
             border:1px solid #333; margin-bottom:0.4rem; }
.card-boosted { border-left: 4px solid #52B788 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data (cached) ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

@st.cache_data
def load_all():
    cat_path = DATA_DIR / "catalogue.csv"
    rs_path  = DATA_DIR / "rec_share.json"

    if cat_path.exists():
        # ── Real data from data_loader.py ──────────────────────────────
        cat = pd.read_csv(cat_path)
        # Ensure genres column is a proper list (CSV stores it as string repr)
        from src.synthetic_data import parse_genres
        cat['genres'] = cat['genres'].apply(parse_genres)
        # Ensure item_id column exists (data_loader uses 'slug' as id)
        if 'item_id' not in cat.columns and 'slug' in cat.columns:
            cat = cat.rename(columns={'slug': 'item_id'})
        # Load real rec_share if available, else compute from catalogue
        if rs_path.exists():
            rec_share_bl = json.loads(rs_path.read_text())
        else:
            obs = generate_observation_sample(cat, n_sessions=200, seed=42)
            rec_share_bl = compute_rec_share(obs)
    else:
        # ── Synthetic fallback ─────────────────────────────────────────
        cat = generate_catalogue(n_items=300, seed=42)
        obs = generate_observation_sample(cat, n_sessions=200, seed=42)
        rec_share_bl = compute_rec_share(obs)

    users     = generate_users(cat, n_users=30, seed=42)
    cat_share = compute_cat_share(cat)
    fm, ids, mlb = build_feature_matrix(cat)
    return cat, users, cat_share, rec_share_bl, fm, ids

cat, users_df, cat_share, rec_share_baseline, feature_matrix, item_ids = load_all()

# ── Pipeline helper ───────────────────────────────────────────────────────────
def run_pipeline(user_profile, lambda_weight, diversity_factor, top_n):
    scored = score_items_for_user(cat, user_profile, feature_matrix, item_ids, DEFAULT_POPULARITY_BIAS)
    scored = apply_user_preferences(scored, user_profile)
    mn, mx = scored['base_score'].min(), scored['base_score'].max()
    scored['base_score'] = (scored['base_score'] - mn) / (mx - mn + 1e-9)
    scored['current_score'] = scored['base_score']
    diverse = rerank_for_diversity(scored.head(top_n * 6), top_n=top_n * 2, diversity_factor=diversity_factor)
    diverse['current_score'] = diverse['base_score']
    final   = rerank_for_fairness(diverse, cat_share, rec_share_baseline, lambda_weight=lambda_weight).head(top_n)
    return scored, final

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📺 NPO Start")
    st.markdown("*Public Values Recommender*")
    st.divider()

    st.markdown("### 👤 Viewer Persona")
    persona = st.selectbox(
        "Select persona", list(NPO_PERSONAS.keys()),
        format_func=lambda x: x.title(),
    )
    st.caption(f"_{NPO_PERSONAS[persona]['description']}_")

    match = users_df[users_df['persona'] == persona]
    user_profile = match.iloc[0].to_dict() if not match.empty else {
        'user_id': 'guest', 'persona': persona,
        'preferred_genres': NPO_PERSONAS[persona]['preferred_genres'],
        'broadcaster_affinity': NPO_PERSONAS[persona]['broadcaster_affinity'],
        'genre_weights': NPO_PERSONAS[persona]['genre_weights'],
        'watch_history': [], 'lambda_preference': 0.5,
        'diversity_preference': 0.4, 'show_explanations': True,
    }
    st.divider()

    st.markdown("### ⚙️ Your Preferences")
    st.caption("*Autonomy — Kiron Putman*")

    lambda_val = st.slider(
        "Fairness weight (λ)", min_value=0.10, max_value=1.00,
        value=float(user_profile.get('lambda_preference', 0.5)), step=0.05,
        help="Higher λ = more fairness correction for underexposed broadcasters. "
             "Minimum 0.10 enforced by Mediawet 2008."
    )
    st.caption("⚖️ Min λ = 0.10  |  Mediawet 2008 floor")

    diversity_val = st.slider(
        "Diversity strength", min_value=0.0, max_value=1.0,
        value=float(user_profile.get('diversity_preference', 0.4)), step=0.05,
        help="Higher value = more genre variety in your list (lower ILS)."
    )
    top_n = st.slider("Recommendations to show", 6, 12, 9, step=3)

    st.divider()
    st.markdown("### 🎭 Genre Preferences")
    all_genres = sorted(set(g for gs in cat['genres'] for g in gs))
    pref_default = [g for g in user_profile.get('preferred_genres', []) if g in all_genres]
    selected_genres = st.multiselect("Preferred genres", all_genres, default=pref_default)
    if selected_genres:
        user_profile['preferred_genres'] = selected_genres
        for g in selected_genres:
            user_profile.setdefault('genre_weights', {})[g] = max(
                user_profile.get('genre_weights', {}).get(g, 0.0), 0.6)
    st.divider()
    show_explanations = st.toggle("Show explanation labels", value=True)
    show_scores       = st.toggle("Show score breakdown", value=False)

# ── Run pipeline ──────────────────────────────────────────────────────────────
scored_df, final_df = run_pipeline(user_profile, lambda_val, diversity_val, top_n)
baseline_top        = scored_df.head(top_n).copy()
baseline_top['fairness_boosted'] = False

eg_before  = compute_exposure_gap(cat_share, rec_share_baseline)
eg_after   = compute_exposure_gap(cat_share, compute_rec_share(final_df))
eg_improve = (eg_before - eg_after) / eg_before * 100 if eg_before > 0 else 0

ils_before = compute_ils(baseline_top.to_dict('records'))
ils_after  = compute_ils(final_df.to_dict('records'))

# ── Header ────────────────────────────────────────────────────────────────────
# Show data source indicator
_cat_path = DATA_DIR / "catalogue.csv"
_data_source = f"🟢 Real data — {len(cat)} NPO series loaded from data/processed/" if _cat_path.exists() else f"🟡 Synthetic data — {len(cat)} items (run `python src/data_loader.py` for real data)"
st.caption(_data_source)

st.markdown('<div class="npo-header">📺 NPO Start — Public Values Recommender System</div>',
            unsafe_allow_html=True)

tab_recs, tab_fair, tab_div, tab_profile, tab_about = st.tabs([
    "🎬 Aanbevolen voor jou",
    "⚖️ Fairness",
    "🌍 Diversity",
    "👤 My Profile",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommendations
# ══════════════════════════════════════════════════════════════════════════════
with tab_recs:
    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exposure Gap — baseline", f"{eg_before:.3f}")
    c2.metric("Exposure Gap — after re-ranking", f"{eg_after:.3f}",
              delta=f"{eg_after - eg_before:+.3f}", delta_color="inverse")
    c3.metric("EG improvement", f"{eg_improve:.0f}%")
    c4.metric("ILS reduction", f"{ils_before - ils_after:.3f}",
              help="Positive = more diverse genre list after re-ranking")
    st.divider()

    # Side-by-side: before vs after
    col_before, col_spacer, col_after = st.columns([5, 1, 5])

    def render_card(item, user_profile, show_exp, show_score, col):
        b = item.get('broadcaster', '')
        colour = BROADCASTER_COLOURS.get(b, '#999')
        boosted = item.get('fairness_boosted', False)
        border = "border-left:4px solid #52B788;" if boosted else ""
        genres = item.get('genres') or []
        genre_chips = "".join([
            f'<span style="background:{colour};color:white;padding:1px 7px;'
            f'border-radius:10px;font-size:0.68rem;margin:1px;display:inline-block">{g}</span>'
            for g in genres
        ])
        boost_badge = '<span style="float:right;font-size:0.65rem;color:#52B788">⬆ Fairness boost</span>' if boosted else ''
        col.markdown(f"""
<div style="background:#1e1e1e;border-radius:8px;padding:0.7rem;border:1px solid #333;{border}margin-bottom:0.3rem">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
    <span style="background:{colour};color:white;font-size:0.65rem;font-weight:700;
                 padding:1px 8px;border-radius:4px">{b}</span>{boost_badge}
  </div>
  <div style="font-size:0.88rem;font-weight:600;color:#f0f0f0;margin-bottom:4px">
    {item.get('title','')}</div>
  <div>{genre_chips}</div>
</div>""", unsafe_allow_html=True)
        if show_exp:
            reason = get_primary_reason(item, user_profile)
            col.caption(reason)
        if show_score:
            details = get_feature_details(item, user_profile)
            with col.expander("Scores"):
                for k, v in details['score_breakdown'].items():
                    st.write(f"**{k}:** {v}")

    with col_before:
        st.markdown("##### 🔴 Before — CTR only (baseline)")
        st.caption(f"EG = {eg_before:.3f}  |  ILS = {ils_before:.3f}")
        for item in baseline_top.to_dict('records'):
            render_card(item, user_profile, show_explanations, show_scores, col_before)

    with col_spacer:
        st.markdown("")

    with col_after:
        st.markdown(f"##### 🟢 After — Fairness + Diversity re-ranking (λ={lambda_val:.2f})")
        st.caption(f"EG = {eg_after:.3f}  |  ILS = {ils_after:.3f}")
        for item in final_df.to_dict('records'):
            render_card(item, user_profile, show_explanations, show_scores, col_after)

    st.divider()

    # Broadcaster share bar — after
    st.markdown("##### Broadcaster share in re-ranked list")
    bc_counts = final_df['broadcaster'].value_counts().reset_index()
    bc_counts.columns = ['Broadcaster', 'Count']
    bc_counts['Share (%)'] = bc_counts['Count'] / bc_counts['Count'].sum() * 100
    fig = px.bar(bc_counts, x='Broadcaster', y='Share (%)', color='Broadcaster',
                 color_discrete_map=BROADCASTER_COLOURS, text_auto='.1f', height=240)
    fig.update_layout(showlegend=False, margin=dict(t=5, b=5),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Fairness
# ══════════════════════════════════════════════════════════════════════════════
with tab_fair:
    st.markdown("### ⚖️ Fairness Dashboard — Producer-side Fairness")
    st.caption("AmanDeep Singh  |  Metric: Exposure Gap (EG)  |  Mediawet 2008")
    st.markdown(
        "**EG = (1/|B|) × Σ |rec\\_share(b) − cat\\_share(b)|**  — lower is fairer."
    )
    st.divider()

    bc_list  = list(BROADCASTER_COLOURS.keys())
    cat_vals = [cat_share.get(b, 0) * 100 for b in bc_list]
    obs_vals = [rec_share_baseline.get(b, 0) * 100 for b in bc_list]
    aft_vals = [compute_rec_share(final_df).get(b, 0) * 100 for b in bc_list]

    cl, cr = st.columns(2)
    with cl:
        st.markdown(f"#### Before  —  EG = **{eg_before:.3f}**")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name='Catalogue share', x=bc_list, y=cat_vals,
                               marker_color='#457B9D'))
        fig1.add_trace(go.Bar(name='Observed rec share', x=bc_list, y=obs_vals,
                               marker_color='#E63946'))
        fig1.update_layout(barmode='group', height=300,
                           legend=dict(orientation='h', y=-0.35),
                           margin=dict(t=5, b=5), yaxis_title='Share (%)',
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

    with cr:
        colour_eg = "#2A9D8F" if eg_after < eg_before else "#E63946"
        st.markdown(f"#### After  —  EG = <span style='color:{colour_eg}'>{eg_after:.3f}</span>", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Catalogue share', x=bc_list, y=cat_vals,
                               marker_color='#457B9D'))
        fig2.add_trace(go.Bar(name='Re-ranked rec share', x=bc_list, y=aft_vals,
                               marker_color='#52B788'))
        fig2.update_layout(barmode='group', height=300,
                           legend=dict(orientation='h', y=-0.35),
                           margin=dict(t=5, b=5), yaxis_title='Share (%)',
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # λ grid search
    st.markdown("#### λ Sensitivity — EG across fairness weights (grid search)")
    st.caption("This is how λ is calibrated: choose the value that minimises EG while preserving engagement.")
    lambda_grid = np.arange(0.0, 1.05, 0.05)
    eg_grid = []
    for lam in lambda_grid:
        _, tmp = run_pipeline(user_profile, float(lam), diversity_val, top_n)
        eg_grid.append(compute_exposure_gap(cat_share, compute_rec_share(tmp)))

    fig_lam = go.Figure()
    fig_lam.add_trace(go.Scatter(x=lambda_grid, y=eg_grid, mode='lines+markers',
                                  line=dict(color='#FF6600', width=2), marker=dict(size=5)))
    fig_lam.add_vline(x=lambda_val, line_dash='dash', line_color='#52B788',
                      annotation_text=f"λ = {lambda_val:.2f}")
    fig_lam.add_hline(y=eg_before, line_dash='dot', line_color='#E63946',
                      annotation_text="Baseline EG")
    fig_lam.update_layout(height=280, xaxis_title='λ', yaxis_title='Exposure Gap (EG)',
                          margin=dict(t=5, b=5),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_lam, use_container_width=True)

    # Fairness correction table
    st.markdown("#### Fairness correction per broadcaster")
    fc_rows = []
    for b in bc_list:
        fc = fairness_correction(b, cat_share, rec_share_baseline)
        fc_rows.append({
            'Broadcaster': b,
            'Catalogue share': f"{cat_share.get(b,0):.1%}",
            'Rec share (baseline)': f"{rec_share_baseline.get(b,0):.1%}",
            'Fairness correction': f"{fc:.3f}",
            'Status': '✅ Underrepresented — boosted' if fc > 0 else '⚠️ Overexposed — not boosted',
        })
    st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)

    st.info(
        "**Note:** EG computed on a small recommendation list (9 items) is inherently noisy. "
        "The metric is most meaningful when aggregated across many users and sessions, "
        "which Assignment 2 will implement. The λ sensitivity chart shows the trend correctly "
        "even if individual EG values fluctuate."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Diversity
# ══════════════════════════════════════════════════════════════════════════════
with tab_div:
    st.markdown("### 🌍 Diversity Dashboard — Content Variety")
    st.caption("Padma Dhuney  |  Metric: Intra-List Similarity (ILS)  |  Jaccard distance on genre tags")
    st.markdown("**ILS = mean pairwise Jaccard similarity across all item pairs** — lower is more diverse.")
    st.divider()

    d1, d2, d3 = st.columns(3)
    d1.metric("ILS — baseline", f"{ils_before:.3f}")
    d2.metric("ILS — after re-ranking", f"{ils_after:.3f}",
              delta=f"{ils_after - ils_before:+.3f}", delta_color="inverse")
    d3.metric("Diversity improvement", f"{(ils_before-ils_after)/ils_before*100:.0f}%" if ils_before > 0 else "—")

    st.divider()

    # Genre distribution comparison
    def genre_freq(df):
        gs = [g for row in df['genres'] for g in (row if isinstance(row, list) else [])]
        s = pd.Series(gs).value_counts()
        return (s / s.sum() * 100).reset_index().rename(columns={'index':'Genre', 0:'Pct', 'count':'Pct', 'proportion':'Pct'})

    gb = genre_freq(baseline_top)
    ga = genre_freq(final_df)
    if 'Genre' not in gb.columns:
        gb.columns = ['Genre', 'Pct']
    if 'Genre' not in ga.columns:
        ga.columns = ['Genre', 'Pct']

    merged = gb.merge(ga, on='Genre', how='outer', suffixes=(' Before', ' After')).fillna(0)

    fig_div = go.Figure()
    fig_div.add_trace(go.Bar(name='Before re-ranking', x=merged['Genre'],
                              y=merged.get('Pct Before', merged.iloc[:,1]),
                              marker_color='#E63946'))
    fig_div.add_trace(go.Bar(name='After re-ranking', x=merged['Genre'],
                              y=merged.get('Pct After', merged.iloc[:,2]),
                              marker_color='#2A9D8F'))
    fig_div.update_layout(barmode='group', height=320, yaxis_title='% of recommendation list',
                          legend=dict(orientation='h', y=-0.3), margin=dict(t=5, b=5),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_div, use_container_width=True)

    # Diversity strength sensitivity
    st.markdown("#### Diversity factor sensitivity")
    div_range = np.arange(0.0, 1.05, 0.1)
    ils_range = []
    for d in div_range:
        sc2 = scored_df.copy()
        sc2['current_score'] = sc2['base_score']
        dv = rerank_for_diversity(sc2.head(top_n*6), top_n=top_n*2, diversity_factor=float(d))
        ils_range.append(compute_ils(dv.head(top_n).to_dict('records')))

    fig_ils = go.Figure()
    fig_ils.add_trace(go.Scatter(x=div_range, y=ils_range, mode='lines+markers',
                                  line=dict(color='#2A9D8F', width=2), marker=dict(size=5)))
    fig_ils.add_vline(x=diversity_val, line_dash='dash', line_color='#F4A261',
                      annotation_text=f"Current = {diversity_val:.2f}")
    fig_ils.update_layout(height=260, xaxis_title='Diversity factor',
                          yaxis_title='Intra-List Similarity (ILS)',
                          margin=dict(t=5, b=5),
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_ils, use_container_width=True)

    st.info(
        "The diversity re-ranker uses a greedy selection algorithm. At each step, it picks the item "
        "that maximises: **selection_score = base_score − diversity_factor × mean_similarity_to_selected**. "
        "A higher diversity factor penalises genre repetition more strongly."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Profile / Autonomy
# ══════════════════════════════════════════════════════════════════════════════
with tab_profile:
    st.markdown("### 👤 My Profile — Autonomy")
    st.caption("Kiron Putman  |  User-controlled recommendation settings")
    st.divider()

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### What the system knows about you")
        st.markdown(f"**Persona:** {persona.title()}")
        st.markdown(f"**Description:** _{NPO_PERSONAS[persona]['description']}_")
        pref_str = ", ".join(user_profile.get('preferred_genres', []))
        st.markdown(f"**Preferred genres:** {pref_str or 'None set'}")
        aff = user_profile.get('broadcaster_affinity', [])
        st.markdown(f"**Broadcaster affinity:** {', '.join(aff) if aff else 'No strong affinity'}")
        hist = user_profile.get('watch_history', [])
        st.markdown(f"**Watch history items:** {len(hist)}")

    with p2:
        st.markdown("#### Current algorithm settings")
        st.markdown(f"**Fairness weight (λ):** `{lambda_val:.2f}`")
        if lambda_val <= 0.15:
            st.warning("λ at Mediawet floor (0.10). Fairness correction is minimal but legally required.")
        elif lambda_val >= 0.70:
            st.success("High λ — smaller broadcasters are strongly boosted.")
        else:
            st.info("Balanced fairness–relevance trade-off.")

        st.markdown(f"**Diversity strength:** `{diversity_val:.2f}`")
        st.markdown(f"**Explanations visible:** `{show_explanations}`")

        st.divider()
        st.markdown("**Design note:** The λ slider gives you control over how much the fairness "
                    "correction affects your recommendations. However, the minimum is fixed at 0.10 "
                    "because NPO is legally required under the Mediawet 2008 to ensure equitable "
                    "broadcaster representation. Your autonomy operates within this public mandate.")

    st.divider()
    if hist:
        st.markdown("#### Your watch history (synthetic)")
        hist_df = cat[cat['item_id'].isin(hist)][['title','broadcaster','genres','publication_date']].copy()
        hist_df['genres'] = hist_df['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### How does the algorithm use your data?")
    st.markdown(get_algorithm_explainer())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About this prototype")
    st.markdown("""
This prototype was built for **INFOMPPM — Personalisation for (Public) Media** at Utrecht University (2025–2026)
as Assignment 2: a working recommender system for NPO Start that integrates four public values.

---

#### Pipeline architecture

```
[1] Content-based scoring
    Cosine similarity on genre feature vectors
    + Popularity bias (simulating NPO Start's CTR-driven baseline)
    + User preference boost (Autonomy — Kiron Putman)
         ↓
[2] Diversity re-ranking  (Padma Dhuney)
    Greedy ILS reduction using Jaccard distance on genre tags
         ↓
[3] Fairness re-ranking   (AmanDeep Singh)
    Broadcaster-aware EG correction with λ-weighted score
    Normalised fairness_correction ensures meaningful signal
    Mediawet 2008 floor: λ ≥ 0.10
         ↓
[4] Explanation labels    (Lisa Wang)
    Human-readable reason on every card + info pop-up
```

---

#### Public values
    """)

    va, vb, vc, vd = st.columns(4)
    va.markdown("**⚖️ Fairness**  \nEquitable broadcaster exposure. EG metric. Mediawet 2008 floor on λ.")
    vb.markdown("**🌍 Diversity**  \nGenre variety via ILS. Greedy re-ranker on Jaccard similarity.")
    vc.markdown("**🔍 Transparency**  \nReason label on every card. Info pop-up with score breakdown.")
    vd.markdown("**🎛️ Autonomy**  \nλ slider + genre preferences + profile page.")

    st.divider()
    st.markdown("""
#### Key design decision — Autonomy vs Fairness floor

Kiron's autonomy slider lets users set λ. However, NPO is governed by the Mediawet 2008,
which mandates balanced broadcaster representation. The system enforces **λ ≥ 0.10**,
ensuring the fairness correction is never fully disabled. Users have agency within
this legal constraint, not over it. This trade-off is documented in the report.

#### Data sources

| Source | Type | Used for |
|---|---|---|
| NPO POMS API schema | Real structure | Catalogue fields (broadcaster, genre, metadata) |
| NPO Start "Aanbevolen voor jou" | Real observation | Biased baseline rec_share |
| Synthetic catalogue | Generated (300 items) | Realistic content across 6 broadcasters |
| Synthetic users | Generated (30 profiles) | 6 NPO-adapted viewer personas |

#### Group members

| Name | Public Value | Module |
|---|---|---|
| AmanDeep Singh | Fairness | `src/fairness.py` |
| Padma Dhuney | Diversity | `src/diversity.py` |
| Lisa Wang | Transparency | `src/transparency.py` |
| Kiron Putman | Autonomy | `src/user_profiles.py` |
    """)
