# src/scoring.py
# Author: AmanDeep Singh
# Base content-based scoring using cosine similarity
# Consistent with Week 5 course materials (Nearest Neighbour)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from src.synthetic_data import parse_genres


def build_feature_matrix(catalogue_df: pd.DataFrame) -> tuple:
    """
    Build a binary feature matrix from genre tags for cosine similarity.
    Consistent with Week 5 approach using sklearn.

    Args:
        catalogue_df: DataFrame with 'item_id' and 'genres' (list of strings)

    Returns:
        (feature_matrix, item_ids, mlb) tuple
    """
    mlb = MultiLabelBinarizer()
    genres = catalogue_df['genres'].apply(parse_genres)
    feature_matrix = mlb.fit_transform(genres)
    item_ids = catalogue_df['item_id'].tolist()
    return feature_matrix, item_ids, mlb


def score_items_for_user(
    catalogue_df: pd.DataFrame,
    user_profile: dict,
    feature_matrix: np.ndarray,
    item_ids: list,
    popularity_bias: dict = None,
) -> pd.DataFrame:
    """
    Score all catalogue items for a given user using cosine similarity
    between the user's watch history and each candidate item.

    Also applies a popularity bias (simulating AVROTROS/MAX dominance)
    to replicate the unfair baseline before intervention.

    Args:
        catalogue_df: full content catalogue
        user_profile: dict with 'watch_history' (list of item_ids)
        feature_matrix: pre-built feature matrix
        item_ids: ordered list of item_ids matching feature_matrix rows
        popularity_bias: dict {broadcaster: bias_weight} to inflate dominant broadcasters

    Returns:
        DataFrame with columns: ['item_id', 'broadcaster', 'genres',
                                  'title', 'base_score', 'recency_score']
    """
    watch_history = user_profile.get('watch_history', [])

    if not watch_history:
        # Cold start: score by recency only
        df = catalogue_df.copy()
        df['base_score'] = np.random.uniform(0.1, 0.5, len(df))
    else:
        # Find indices of watched items
        watched_indices = [
            item_ids.index(iid) for iid in watch_history if iid in item_ids
        ]

        if not watched_indices:
            df = catalogue_df.copy()
            df['base_score'] = np.random.uniform(0.1, 0.5, len(df))
        else:
            # User vector = mean of watched item vectors
            user_vector = feature_matrix[watched_indices].mean(axis=0).reshape(1, -1)
            scores = cosine_similarity(user_vector, feature_matrix).flatten()

            df = catalogue_df.copy()
            df['base_score'] = scores

    # Recency score
    df['recency_score'] = _compute_recency(df)

    # Apply popularity bias to simulate the unfair baseline
    if popularity_bias:
        df['base_score'] = df.apply(
            lambda row: row['base_score'] * (1 + popularity_bias.get(row['broadcaster'], 0)),
            axis=1
        )

    # Exclude already watched items
    df = df[~df['item_id'].isin(watch_history)]

    return df.sort_values('base_score', ascending=False).reset_index(drop=True)


def _compute_recency(df: pd.DataFrame) -> pd.Series:
    """Normalised recency score based on publication_date."""
    if 'publication_date' not in df.columns:
        return pd.Series(np.random.uniform(0.1, 1.0, len(df)), index=df.index)

    dates = pd.to_datetime(df['publication_date'], errors='coerce')
    if dates.isna().all():
        return pd.Series(np.random.uniform(0.1, 1.0, len(df)), index=df.index)

    min_date = dates.min()
    max_date = dates.max()
    if min_date == max_date:
        return pd.Series(1.0, index=df.index)

    return (dates - min_date) / (max_date - min_date)


# Default popularity bias to simulate NPO Start's CTR-driven imbalance
DEFAULT_POPULARITY_BIAS = {
    "AVROTROS": 0.5,   # Dominant broadcaster · inflated
    "MAX": 0.4,         # Large catalogue · inflated
    "KRO-NCRV": 0.2,
    "VPRO": 0.0,        # No inflation · structurally disadvantaged
    "NTR": 0.0,
    "EO": 0.0,
}
