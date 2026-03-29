# src/diversity.py
# Code Implemented by AmanDeep Singh
# Public Value: Diversity — content variety for users
# Implements diversity-aware re-ranking using Intra-List Similarity (ILS).
# Lower ILS = more diverse recommendation list.
# Uses Jaccard distance on genre tags as the similarity measure,
# consistent with the course materials (Week 3).

import numpy as np
import pandas as pd
from itertools import combinations
from src.synthetic_data import parse_genres


def jaccard_similarity(genres_a: set, genres_b: set) -> float:
    """
    Compute Jaccard similarity between two sets of genre tags.

    jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0 if both sets are empty.
    """
    if not genres_a and not genres_b:
        return 0.0
    intersection = len(genres_a & genres_b)
    union = len(genres_a | genres_b)
    return intersection / union if union > 0 else 0.0


def compute_ils(items: list[dict], genre_key: str = 'genres') -> float:
    """
    Compute Intra-List Similarity (ILS) for a list of recommended items.
    ILS = mean pairwise Jaccard similarity across all item pairs.

    Lower ILS = more diverse list.

    Args:
        items: list of dicts, each with a genre_key field containing a set/list of genres
        genre_key: the key in each dict that holds genre tags

    Returns:
        float: ILS score in [0, 1]
    """
    if len(items) < 2:
        return 0.0

    pairs = list(combinations(items, 2))
    similarities = [
        jaccard_similarity(set(parse_genres(a[genre_key])), set(parse_genres(b[genre_key])))
        for a, b in pairs
    ]
    return np.mean(similarities)


def rerank_for_diversity(
    candidate_df: pd.DataFrame,
    top_n: int = 10,
    diversity_factor: float = 0.4,
    genre_key: str = 'genres',
) -> pd.DataFrame:
    """
    Apply diversity-aware re-ranking using a greedy selection strategy.

    For each position in the top-N list, selects the item that maximises:
        selection_score = current_score − diversity_factor × mean_similarity_to_already_selected

    This penalises items that are too similar to already selected items,
    promoting genre variety across the recommendation list.

    Args:
        candidate_df: DataFrame with columns ['item_id', genre_key, 'current_score']
        top_n: number of items to return
        diversity_factor: how strongly to penalise similarity (0 = no diversity, 1 = max diversity)
        genre_key: column name holding genre tag lists

    Returns:
        DataFrame of top_n items re-ranked for diversity, with added column:
        'diversity_penalised' (bool — True if item was moved down due to diversity)
    """
    df = candidate_df.copy().reset_index(drop=True)
    original_order = df['item_id'].tolist()

    selected = []
    remaining = df.to_dict(orient='records')

    while len(selected) < top_n and remaining:
        best_item = None
        best_score = -np.inf

        for item in remaining:
            if not selected:
                adjusted_score = item['current_score']
            else:
                mean_sim = np.mean([
                    jaccard_similarity(set(parse_genres(item[genre_key])), set(parse_genres(s[genre_key])))
                    for s in selected
                ])
                adjusted_score = item['current_score'] - diversity_factor * mean_sim

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_item = item

        selected.append(best_item)
        remaining.remove(best_item)

    result_df = pd.DataFrame(selected)
    result_df['diversity_penalised'] = ~result_df['item_id'].isin(
        original_order[:top_n]
    )
    return result_df.reset_index(drop=True)
