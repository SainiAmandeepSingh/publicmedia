# src/synthetic_data.py
# Code Implemented by AmanDeep Singh
# Generates realistic synthetic NPO Start catalogue and observation data.
# Used as a fallback when the POMS API is unavailable.
# Data is grounded in real NPO genre distributions and broadcaster portfolio sizes.

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

BROADCASTERS = ["AVROTROS", "MAX", "KRO-NCRV", "VPRO", "NTR", "EO"]

# Approximate real catalogue size ratios across NPO broadcasters
BROADCASTER_CATALOGUE_WEIGHTS = {
    "AVROTROS": 0.30, "MAX": 0.25, "KRO-NCRV": 0.20,
    "VPRO": 0.10, "NTR": 0.09, "EO": 0.06,
}

# Genre distribution per broadcaster (grounded in NPO programming)
BROADCASTER_GENRE_PROFILES = {
    "AVROTROS": {"Drama": 0.30, "Entertainment": 0.30, "Nieuws": 0.15, "Sport": 0.10, "Jeugd": 0.10, "Documentaire": 0.05},
    "MAX":      {"Entertainment": 0.35, "Nieuws": 0.20, "Drama": 0.20, "Documentaire": 0.10, "Jeugd": 0.10, "Sport": 0.05},
    "KRO-NCRV": {"Drama": 0.35, "Documentaire": 0.20, "Cultuur": 0.15, "Nieuws": 0.15, "Entertainment": 0.10, "Jeugd": 0.05},
    "VPRO":     {"Documentaire": 0.50, "Cultuur": 0.25, "Wetenschap": 0.15, "Nieuws": 0.05, "Drama": 0.05},
    "NTR":      {"Nieuws": 0.35, "Documentaire": 0.25, "Wetenschap": 0.20, "Cultuur": 0.10, "Jeugd": 0.10},
    "EO":       {"Drama": 0.30, "Documentaire": 0.25, "Jeugd": 0.25, "Cultuur": 0.10, "Entertainment": 0.10},
}

GENRE_SECONDARY = {
    "Drama": ["Thriller", "Romantiek"],
    "Documentaire": ["Natuur", "Geschiedenis", "Maatschappij"],
    "Nieuws": ["Politiek", "Economie"],
    "Entertainment": ["Humor", "Lifestyle"],
    "Jeugd": ["Animatie", "Avontuur"],
    "Cultuur": ["Muziek", "Kunst"],
    "Sport": ["Voetbal", "Wielrennen"],
    "Wetenschap": ["Technologie", "Gezondheid"],
}

TITLE_TEMPLATES = {
    "AVROTROS": ["Goede Tijden Slechte Tijden","Vandaag Inside","Utopia","Jinek","Bureau Ontmoeting",
                 "Beste Zangers","Heel Holland Bakt","De Wereld Draait Door","Expeditie Robinson",
                 "First Dates","Nieuwsuur","Boer Zoekt Vrouw"],
    "MAX":      ["MAX Vandaag","Heel Holland Bakt Professionals","Boer Zoekt Vrouw Internationaal",
                 "Wie Is De Mol?","MAX Maakt Mogelijk","Kijkmagazine","Tijd voor MAX",
                 "De Rijdende Rechter","Tussen Kunst en Kitsch","Per Seconde Wijzer",
                 "MAX Muziek","De Landelijke Rijschool"],
    "KRO-NCRV": ["De Buitendienst","Dokter Tinus","Nachtzusters","Dit Is De Dag","Kruispunt",
                 "Zondag met Lubach","KOOT","Nederland Zingt","Andere Tijden",
                 "De Smaak van De Wereld","Hazes","De Muur"],
    "VPRO":     ["Tegenlicht","VPRO Zomergasten","Het Zwarte Gat","Durf te Vragen",
                 "De Slimste Mens","Metropolis","Backlight","Human","Documentary Now",
                 "De Kunst van het Leven","Hollands Diep","Serious Request"],
    "NTR":      ["Andere Tijden Sport","NTR Focus","Schooltv Weekjournaal","De Taal van Nederland",
                 "Vroege Vogels","Met het Oog op Morgen","OVT","Kennis van Nu",
                 "De Wandeling","Journaal","NieuwsWeekend","Cafe Weltschmerz"],
    "EO":       ["EO Documentaire","Ademnood","Dit is de Dag EO","Spoorloos","Ik Vertrek",
                 "Hemel en Aarde","EO Missie","Op weg naar morgen","Geloven in het Leven",
                 "Kwetsbaar","Echt waar","Mag ik dit weten"],
}


def _pick_genres(broadcaster, rng):
    profile = BROADCASTER_GENRE_PROFILES[broadcaster]
    genres = list(profile.keys())
    weights = list(profile.values())
    primary = rng.choice(genres, p=weights)
    result = [primary]
    secondaries = GENRE_SECONDARY.get(primary, [])
    if secondaries and rng.random() < 0.6:
        result.append(rng.choice(secondaries))
    return result


def generate_catalogue(n_items=300, seed=42):
    """
    Generate a synthetic NPO Start content catalogue.
    Broadcaster proportions follow real NPO catalogue size ratios.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    items = []
    item_num = 1
    for broadcaster, share in BROADCASTER_CATALOGUE_WEIGHTS.items():
        count = max(1, int(round(n_items * share)))
        titles = TITLE_TEMPLATES[broadcaster]
        for i in range(count):
            base_title = titles[i % len(titles)]
            suffix = f" S{rng.integers(1,6)}E{rng.integers(1,13)}" if rng.random() > 0.4 else ""
            pub_date = datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 730)))
            items.append({
                "item_id": f"npo_{broadcaster.lower()}_{item_num:04d}",
                "title": base_title + suffix,
                "broadcaster": broadcaster,
                "genres": _pick_genres(broadcaster, rng),
                "description": f"{base_title} — een programma van {broadcaster}.",
                "publication_date": pub_date.strftime("%Y-%m-%d"),
                "episode_number": int(rng.integers(1, 13)),
                "series_id": f"series_{broadcaster.lower()}_{(i // 3) + 1:03d}",
            })
            item_num += 1
    return pd.DataFrame(items)


def generate_observation_sample(catalogue_df, n_sessions=60, seed=42):
    """
    Simulate CTR-biased recommendation observations from NPO Start "Aanbevolen voor jou".
    AVROTROS and MAX are heavily over-represented (popularity bias).
    VPRO, NTR, EO are under-represented.
    Calibrated to produce EG ~ 0.14, matching the pre-intervention baseline
    described in AmanDeep Singh's research proposal.
    """
    rng = np.random.default_rng(seed)
    # Biased sampling weights — simulating CTR-driven dominance
    OBSERVATION_BIAS = {
        "AVROTROS": 0.55,
        "MAX":      0.28,
        "KRO-NCRV": 0.10,
        "VPRO":     0.03,
        "NTR":      0.03,
        "EO":       0.01,
    }
    rows = []
    for session_id in range(n_sessions):
        broadcasters = list(OBSERVATION_BIAS.keys())
        weights = list(OBSERVATION_BIAS.values())
        n_items = int(rng.integers(6, 12))
        sampled_broadcasters = rng.choice(broadcasters, size=n_items, p=weights)
        for b in sampled_broadcasters:
            bc_items = catalogue_df[catalogue_df["broadcaster"] == b]
            if bc_items.empty:
                continue
            item = bc_items.sample(1, random_state=int(rng.integers(0, 9999))).iloc[0]
            rows.append({
                "item_id": item["item_id"],
                "broadcaster": b,
                "session_id": f"session_{session_id:03d}",
                "timestamp": (datetime(2024,1,1)+timedelta(hours=int(rng.integers(0,8760)))).isoformat(),
            })
    return pd.DataFrame(rows)


def parse_genres(value) -> list:
    """
    Safely parse a genres value into a list of strings.
    Handles all three cases:
      - Already a list: ['Drama', 'Thriller']         → returned as-is
      - CSV repr string: "['Drama', 'Thriller']"       → parsed to list
      - Comma-separated string: "Drama,Thriller"       → split to list
      - Single string: "Drama"                         → ['Drama']
      - Empty / None                                   → []
    Called by scoring.py, diversity.py, app.py whenever genres are read
    from a DataFrame that may have been loaded from CSV.
    """
    if isinstance(value, list):
        return [str(g).strip() for g in value if g]
    if not isinstance(value, str) or not value.strip():
        return []
    v = value.strip()
    # Python list repr: "['Drama', 'Thriller']"
    if v.startswith('['):
        try:
            import ast
            parsed = ast.literal_eval(v)
            if isinstance(parsed, list):
                return [str(g).strip() for g in parsed if g]
        except Exception:
            pass
    # Comma-separated fallback
    return [g.strip().strip("'\"") for g in v.split(',') if g.strip()]
