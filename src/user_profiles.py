# src/user_profiles.py
# Owner: Kiron Putman
# Public Value: Autonomy — user control over recommendations
#
# Generates synthetic user profiles grounded in real NPO genre distributions.
# Based on the course's synthesize_user_data.ipynb utility (Utilities folder).
# Personas are adapted from generic archetypes to NPO-specific viewing patterns.

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── NPO-adapted personas ────────────────────────────────────────────────────
# Derived from real POMS genre distribution.
# Each persona maps to NPO genre tags and implicitly to broadcaster affinities.

NPO_PERSONAS = {
    "documentary nerd": {
        "preferred_genres": ["Documentaire", "Cultuur"],
        "broadcaster_affinity": ["VPRO", "NTR"],
        "genre_weights": {
            "Documentaire": 0.70, "Cultuur": 0.15, "Nieuws": 0.05,
            "Drama": 0.03, "Entertainment": 0.02, "Sport": 0.01,
            "Jeugd": 0.02, "Wetenschap": 0.02,
        },
        "description": "Watches primarily documentaries and cultural programmes",
    },
    "news watcher": {
        "preferred_genres": ["Nieuws", "Politiek"],
        "broadcaster_affinity": ["NTR", "AVROTROS"],
        "genre_weights": {
            "Nieuws": 0.65, "Politiek": 0.15, "Documentaire": 0.08,
            "Drama": 0.04, "Entertainment": 0.04, "Sport": 0.02,
            "Cultuur": 0.01, "Jeugd": 0.01,
        },
        "description": "Primarily consumes news and current affairs",
    },
    "drama fan": {
        "preferred_genres": ["Drama", "Thriller"],
        "broadcaster_affinity": ["AVROTROS", "KRO-NCRV"],
        "genre_weights": {
            "Drama": 0.55, "Thriller": 0.20, "Entertainment": 0.10,
            "Nieuws": 0.05, "Documentaire": 0.05, "Cultuur": 0.03,
            "Sport": 0.01, "Jeugd": 0.01,
        },
        "description": "Enjoys drama series and thrillers",
    },
    "family viewer": {
        "preferred_genres": ["Jeugd", "Entertainment"],
        "broadcaster_affinity": ["MAX", "AVROTROS", "EO"],
        "genre_weights": {
            "Jeugd": 0.40, "Entertainment": 0.25, "Drama": 0.10,
            "Sport": 0.08, "Nieuws": 0.07, "Documentaire": 0.05,
            "Cultuur": 0.03, "Wetenschap": 0.02,
        },
        "description": "Watches family and children's content",
    },
    "varied consumer": {
        "preferred_genres": ["Drama", "Nieuws", "Documentaire", "Entertainment"],
        "broadcaster_affinity": [],  # no strong affinity
        "genre_weights": {
            "Drama": 0.18, "Nieuws": 0.18, "Documentaire": 0.16,
            "Entertainment": 0.16, "Cultuur": 0.10, "Sport": 0.08,
            "Jeugd": 0.08, "Wetenschap": 0.06,
        },
        "description": "Watches a broad mix of content across genres",
    },
    "sport fan": {
        "preferred_genres": ["Sport"],
        "broadcaster_affinity": ["MAX", "NOS"],
        "genre_weights": {
            "Sport": 0.75, "Nieuws": 0.10, "Entertainment": 0.07,
            "Drama": 0.04, "Documentaire": 0.02, "Cultuur": 0.01,
            "Jeugd": 0.01,
        },
        "description": "Primarily watches sports content",
    },
}

# Persona distribution reflecting a plausible NPO user base
PERSONA_DISTRIBUTION = {
    "documentary nerd": 10,
    "news watcher": 20,
    "drama fan": 25,
    "family viewer": 15,
    "varied consumer": 20,
    "sport fan": 10,
}


def generate_users(
    catalogue_df: pd.DataFrame,
    n_users: int = 30,
    persona_distribution: dict = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic user profiles grounded in real NPO content distribution.

    Each user gets:
    - user_id, persona, preferred_genres, broadcaster_affinity
    - watch_history: list of item_ids from catalogue matching their persona
    - lambda_preference: initial λ value (autonomy setting)
    - genre_weights: dict for scoring

    Args:
        catalogue_df: real POMS catalogue DataFrame with 'item_id', 'genres', 'broadcaster'
        n_users: number of users to generate
        persona_distribution: % per persona (must sum to 100)
        seed: random seed for reproducibility

    Returns:
        DataFrame of user profiles
    """
    random.seed(seed)
    np.random.seed(seed)

    if persona_distribution is None:
        persona_distribution = PERSONA_DISTRIBUTION

    users = []

    for persona, pct in persona_distribution.items():
        n = int((pct / 100) * n_users)
        persona_config = NPO_PERSONAS[persona]

        for i in range(n):
            # Build watch history from real catalogue items
            watch_history = _sample_watch_history(catalogue_df, persona_config)

            user = {
                "user_id": f"user_{persona.replace(' ', '_')}_{i+1:02d}",
                "persona": persona,
                "persona_description": persona_config["description"],
                "preferred_genres": persona_config["preferred_genres"].copy(),
                "broadcaster_affinity": persona_config["broadcaster_affinity"].copy(),
                "genre_weights": persona_config["genre_weights"].copy(),
                "watch_history": watch_history,
                # Autonomy settings — user can modify these in the interface
                "lambda_preference": round(random.uniform(0.3, 0.7), 2),
                "diversity_preference": round(random.uniform(0.2, 0.6), 2),
                "show_explanations": True,
            }
            users.append(user)

    return pd.DataFrame(users)


def _sample_watch_history(catalogue_df: pd.DataFrame, persona_config: dict, n: int = 10) -> list:
    """Sample n item_ids from catalogue weighted by persona genre preferences."""
    weights = persona_config["genre_weights"]

    def score_item(row):
        genres = row.get('genres', [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(',')]
        return max([weights.get(g, 0.01) for g in genres] or [0.01])

    if 'genres' not in catalogue_df.columns or catalogue_df.empty:
        return []

    df = catalogue_df.copy()
    df['_weight'] = df.apply(score_item, axis=1)
    total = df['_weight'].sum()
    if total == 0:
        return []

    sampled = df.sample(
        n=min(n, len(df)),
        weights=df['_weight'] / total,
        replace=False,
    )
    return sampled['item_id'].tolist()


# ── Autonomy: apply user preference overrides ───────────────────────────────

def apply_user_preferences(
    candidate_df: pd.DataFrame,
    user_profile: dict,
) -> pd.DataFrame:
    """
    Apply user's explicit genre preferences to adjust base scores.
    This represents Kiron's autonomy feature: user preferences directly
    influence the recommendation pipeline.

    Preferred genres receive a score boost; disliked genres receive a penalty.

    Args:
        candidate_df: DataFrame with 'genres' and 'base_score' columns
        user_profile: dict with 'preferred_genres', 'genre_weights'

    Returns:
        DataFrame with updated 'base_score' incorporating user preferences
    """
    df = candidate_df.copy()
    genre_weights = user_profile.get('genre_weights', {})

    def preference_boost(row):
        genres = row.get('genres', [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(',')]
        boost = max([genre_weights.get(g, 0.0) for g in genres] or [0.0])
        return boost

    df['preference_boost'] = df.apply(preference_boost, axis=1)
    df['base_score'] = df['base_score'] + 0.3 * df['preference_boost']
    return df
