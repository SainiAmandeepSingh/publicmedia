# src/fairness.py
# Author: AmanDeep Singh
# Public Value: Fairness · equitable broadcaster exposure
#
# Implements the broadcaster-aware re-ranking algorithm from the research proposal.
# Measures and corrects the Exposure Gap (EG) between broadcaster catalogue share
# and recommendation share.

import numpy as np
import pandas as pd


def compute_cat_share(catalogue_df: pd.DataFrame) -> dict:
    """
    Compute each broadcaster's share of the total content catalogue.

    Args:
        catalogue_df: DataFrame with at minimum a 'broadcaster' column.

    Returns:
        dict: {broadcaster: catalogue_share}
    """
    counts = catalogue_df['broadcaster'].value_counts()
    return (counts / counts.sum()).to_dict()


def compute_rec_share(recommended_df: pd.DataFrame) -> dict:
    """
    Compute each broadcaster's share of the current recommendation output.
    Used as the pre-intervention baseline (observed from NPO Start sampling).

    Args:
        recommended_df: DataFrame with a 'broadcaster' column representing
                        the items currently surfaced in "Aanbevolen voor jou".

    Returns:
        dict: {broadcaster: recommendation_share}
    """
    counts = recommended_df['broadcaster'].value_counts()
    return {str(k): float(v) for k, v in (counts / counts.sum()).items()}


def compute_exposure_gap(cat_share: dict, rec_share: dict) -> float:
    """
    Compute the Exposure Gap (EG) metric.
    EG = (1/|B|) × Σ |rec_share(b) − cat_share(b)|

    A lower EG value indicates greater fairness.

    Args:
        cat_share: dict of broadcaster → catalogue share
        rec_share: dict of broadcaster → recommendation share

    Returns:
        float: EG value in [0, 1]
    """
    broadcasters = set(cat_share.keys()) | set(rec_share.keys())
    gaps = [abs(rec_share.get(b, 0) - cat_share.get(b, 0)) for b in broadcasters]
    return np.mean(gaps)


def fairness_correction(broadcaster: str, cat_share: dict, rec_share: dict) -> float:
    """
    Compute the fairness correction score for a single broadcaster.
    fairness_correction(b) = max(0, cat_share(b) − rec_share(b))

    Underrepresented broadcasters receive a positive correction.
    Overexposed broadcasters receive 0.

    Args:
        broadcaster: broadcaster name string
        cat_share: dict of broadcaster → catalogue share
        rec_share: dict of broadcaster → recommendation share

    Returns:
        float: fairness correction score ≥ 0
    """
    return max(0, cat_share.get(broadcaster, 0) - rec_share.get(broadcaster, 0))


def rerank_for_fairness(
    candidate_df: pd.DataFrame,
    cat_share: dict,
    rec_share: dict,
    lambda_weight: float = 0.5,
    lambda_min: float = 0.1,  # Mediawet 2008 floor · cannot be set to 0
) -> pd.DataFrame:
    """
    Apply fairness re-ranking to a candidate list.

    final_score = (1 − λ) × current_score + λ × fairness_correction(broadcaster)

    Args:
        candidate_df: DataFrame with columns ['item_id', 'broadcaster', 'current_score']
        cat_share: broadcaster catalogue shares
        rec_share: broadcaster recommendation shares (baseline)
        lambda_weight: fairness weight λ ∈ [lambda_min, 1]
        lambda_min: minimum λ enforced by Mediawet 2008 constraint

    Returns:
        DataFrame sorted by final_score descending, with added columns:
        'fairness_correction', 'final_score', 'fairness_boosted' (bool)
    """
    lambda_weight = max(lambda_weight, lambda_min)

    df = candidate_df.copy()
    df['fairness_correction'] = df['broadcaster'].apply(
        lambda b: fairness_correction(b, cat_share, rec_share)
    )
    # Normalise fairness_correction to [0,1] so it's on the same scale as current_score.
    # Without this, a correction of 0.06 cannot compete with a base_score of 1.0.
    fc_max = df['fairness_correction'].max()
    if fc_max > 0:
        df['fairness_correction_norm'] = df['fairness_correction'] / fc_max
    else:
        df['fairness_correction_norm'] = 0.0
    df['final_score'] = (
        (1 - lambda_weight) * df['current_score']
        + lambda_weight * df['fairness_correction_norm']
    )
    df['fairness_boosted'] = df['fairness_correction'] > 0

    return df.sort_values('final_score', ascending=False).reset_index(drop=True)
