# src/data_loader.py
# Handles all data collection: POMS API + observation sampling

import requests
import pandas as pd
import json
import os
from pathlib import Path

POMS_API_BASE = "https://api.poms.omroep.nl"
DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_poms_catalogue(
    broadcasters: list = None,
    max_items: int = 500,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Fetch content catalogue from the NPO POMS API.
    No authentication required — public endpoint.

    Args:
        broadcasters: list of broadcaster codes to filter (e.g. ['VPRO', 'NTR'])
                      If None, fetches all NPO broadcasters.
        max_items: maximum number of items to retrieve
        save_path: optional path to save raw JSON response

    Returns:
        DataFrame with columns:
        ['item_id', 'title', 'broadcaster', 'genres', 'description',
         'publication_date', 'episode_number', 'series_id']
    """
    if broadcasters is None:
        broadcasters = ['VPRO', 'NTR', 'MAX', 'AVROTROS', 'EO', 'KRO-NCRV']

    all_items = []

    for broadcaster in broadcasters:
        url = f"{POMS_API_BASE}/media"
        params = {
            "broadcaster": broadcaster,
            "type": "BROADCAST,SERIES,CLIP",
            "max": min(100, max_items // len(broadcasters)),
            "offset": 0,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                for item in items:
                    all_items.append(_parse_poms_item(item, broadcaster))
            else:
                print(f"Warning: POMS API returned {response.status_code} for {broadcaster}")
        except requests.RequestException as e:
            print(f"Warning: Could not fetch {broadcaster}: {e}")

    if not all_items:
        print("No items fetched from POMS API. Loading fallback data if available.")
        return _load_fallback_catalogue()

    df = pd.DataFrame(all_items)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


def _parse_poms_item(item: dict, broadcaster: str) -> dict:
    """Parse a single POMS API item into a flat dict."""
    genres = []
    for g in item.get("genres", []):
        if isinstance(g, dict):
            genres.append(g.get("value", ""))
        elif isinstance(g, str):
            genres.append(g)

    return {
        "item_id": item.get("mid", ""),
        "title": item.get("title", ""),
        "broadcaster": broadcaster,
        "genres": genres,
        "description": item.get("shortDescription", ""),
        "publication_date": item.get("publishStart", ""),
        "episode_number": item.get("episodeNumber", None),
        "series_id": item.get("seriesRef", {}).get("mid", "") if item.get("seriesRef") else "",
    }


def load_observation_data(path: str = None) -> pd.DataFrame:
    """
    Load pre-collected NPO Start observation data.
    This is the baseline rec_share measured from "Aanbevolen voor jou" sampling.

    Expected CSV columns: ['item_id', 'broadcaster', 'session_id', 'timestamp']

    Args:
        path: path to observation CSV. Defaults to data/raw/observations.csv

    Returns:
        DataFrame of observed recommendations
    """
    if path is None:
        path = DATA_DIR / "raw" / "observations.csv"

    if not os.path.exists(path):
        print(f"Observation data not found at {path}.")
        print("Please run notebooks/01_data_collection.ipynb first.")
        return pd.DataFrame(columns=['item_id', 'broadcaster', 'session_id'])

    return pd.read_csv(path)


def _load_fallback_catalogue() -> pd.DataFrame:
    """
    Load fallback catalogue from data/processed/ if POMS API is unavailable.
    Returns empty DataFrame if no fallback exists.
    """
    fallback = DATA_DIR / "processed" / "catalogue.csv"
    if os.path.exists(fallback):
        print(f"Loading fallback catalogue from {fallback}")
        df = pd.read_csv(fallback)
        # Ensure genres column is parsed as list
        if 'genres' in df.columns:
            df['genres'] = df['genres'].apply(
                lambda x: x.split(',') if isinstance(x, str) else []
            )
        return df
    return pd.DataFrame()
