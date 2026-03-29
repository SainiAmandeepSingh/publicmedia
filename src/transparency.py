# src/transparency.py
# Owner:AmanDeep Singh & Lisa Wang
# Public Value: Transparency — explanations in the interface
#
# Generates human-readable explanation labels for each recommended item.
# These labels appear in the Streamlit interface as per Lisa's proposal:
# - A short reason label on each card
# - An info button pop-up with feature details
# - A "Why does the algorithm recommend things?" general explainer

import pandas as pd


# ── Primary reason labels (short, shown directly on card) ──────────────────

def get_primary_reason(item: dict, user_profile: dict) -> str:
    """
    Generate a short primary reason label for a recommended item.

    Priority order:
    1. Fairness boost (if item was boosted for broadcaster equity)
    2. Genre match with user preference
    3. Watch history match
    4. Diversity pick (if item was selected for variety)
    5. Fallback: popular content

    Args:
        item: dict with keys 'fairness_boosted', 'diversity_penalised',
              'broadcaster', 'genres', 'title'
        user_profile: dict with 'preferred_genres', 'watch_history'

    Returns:
        str: short explanation label
    """
    preferred = set(user_profile.get('preferred_genres', []))
    item_genres = set(item.get('genres', []))

    if item.get('fairness_boosted'):
        broadcaster = item.get('broadcaster', 'this broadcaster')
        return f"🟢 {broadcaster} · smaller broadcaster gaining visibility"

    if item_genres & preferred:
        matching = list(item_genres & preferred)
        return f"📺 Matches your interest in {matching[0]}"

    if item.get('item_id') in user_profile.get('watch_history', []):
        return "🔁 Similar to content you've watched"

    if item.get('diversity_penalised') is False:
        genre = list(item_genres)[0] if item_genres else 'this genre'
        return f"🌍 Broadening your view · {genre} content"

    return "⭐ Popular on NPO Start"


# ── Detailed feature pop-up (shown when user clicks ℹ️) ────────────────────

def get_feature_details(item: dict, user_profile: dict) -> dict:
    """
    Generate the detailed explanation shown in the info pop-up.

    Returns a dict with:
    - 'features': list of feature strings shown as bullet points
    - 'score_breakdown': dict with score components
    - 'data_used': list of data types that influenced this recommendation
    """
    preferred = set(user_profile.get('preferred_genres', []))
    item_genres = set(item.get('genres', []))

    features = []

    genre_match = item_genres & preferred
    if genre_match:
        features.append(f"Matches your interest in {', '.join(genre_match)}")

    if item.get('fairness_boosted'):
        features.append(f"Boosted to give more visibility to {item.get('broadcaster')}")

    if item.get('diversity_penalised') is False and not genre_match:
        features.append("Selected to increase variety in your recommendations")

    recency = item.get('recency_score', None)
    if recency and recency > 0.7:
        features.append("Recently published content")

    if not features:
        features.append("Popular on NPO Start")

    score_breakdown = {
        'Relevance score': round(item.get('base_score', 0), 3),
        'Fairness correction': round(item.get('fairness_correction', 0), 3),
        'Final score': round(item.get('final_score', 0), 3),
    }

    data_used = ['Viewing history', 'Content genre tags', 'Broadcaster catalogue data']
    if item.get('fairness_boosted'):
        data_used.append('Broadcaster exposure statistics')

    return {
        'features': features,
        'score_breakdown': score_breakdown,
        'data_used': data_used,
    }


# ── General algorithm explainer (shown when user clicks ❓) ────────────────

ALGORITHM_EXPLAINER = """
**How does NPO Start recommend content?**

NPO Start's recommender system is built on four public values.

1. **Relevance:** Content is matched to your viewing history and genre preferences. The more you engage, the more personalised your list becomes.

2. **Diversity:** Your list is adjusted to include a range of genres so you are not shown only one type of content.

3. **Fairness:** Smaller broadcasters such as VPRO, NTR, and EO are boosted when underrepresented relative to their catalogue share, as required by the Mediawet 2008.

4. **Your control:** You can adjust the fairness weight, set preferred genres, and control how strongly the system personalises for you using the sidebar.
"""


def get_algorithm_explainer() -> str:
    """Return the general algorithm explanation string."""
    return ALGORITHM_EXPLAINER
