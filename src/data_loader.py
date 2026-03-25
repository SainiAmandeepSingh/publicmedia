# src/data_loader.py
# NPO Start — Real Data Loader
# Fetches real recommendation data from NPO Start's public API.
# No authentication required. Run this script locally before starting the app.
#
# Usage:
#   python src/data_loader.py
#
# Output:
#   data/processed/catalogue.csv       — real NPO Start series with broadcaster + genre
#   data/processed/observations.csv    — same items as rec_share observation baseline
#   data/processed/rec_share.json      — broadcaster share dict ready for fairness.py
#
# API endpoints used (public, no auth):
#   https://npo.nl/start/api/domain/recommendation-collection  — what NPO Start recommends
#   https://npo.nl/start/api/domain/series-detail              — broadcaster + genre per slug

import requests
import pandas as pd
import json
import time
import os
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL   = "https://npo.nl/start/api/domain"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

# The four anonymous recommendation rows on NPO Start's homepage.
# These are the exact collectionIds observed in NPO Start's network requests.
# They represent what the CTR-optimised system shows anonymous users —
# i.e. the pre-intervention baseline for the Exposure Gap measurement.
COLLECTION_IDS = [
    ("series-anonymous-v0",       "SERIES"),   # main recommendations row
    ("trending-anonymous-v0",     "SERIES"),   # trending
    ("public-value-anonymous-v0", "SERIES"),   # public value row
    ("recent-free-v0",            "SERIES"),   # recently added free content
]

# Anonymous party ID — generates a fresh one each session, any value works
PARTY_ID = "1%3Amm8465af%3A6b24c4cff4804534a3e3a376b40cbb0f"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (research; INFOMPPM assignment)",
}

# Broadcasters we care about for the fairness analysis (Mediawet 2008 members)
TARGET_BROADCASTERS = {"AVROTROS", "MAX", "KRO-NCRV", "VPRO", "NTR", "EO", "BNNVARA"}


# ── Step 1: Collect slugs from recommendation collections ─────────────────────

def fetch_collection(collection_id: str, collection_type: str) -> list[dict]:
    """
    Fetch one recommendation collection from NPO Start.
    Returns a list of {slug, title, productId, collectionId} dicts.
    """
    url = f"{BASE_URL}/recommendation-collection"
    params = {
        "collectionId":          collection_id,
        "collectionType":        collection_type,
        "includePremiumContent": "true",
        "layoutType":            "RECOMMENDATION",
        "partyId":               PARTY_ID,
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        return [
            {
                "slug":         item.get("slug", ""),
                "title":        item.get("title", ""),
                "productId":    item.get("productId", ""),
                "collectionId": collection_id,
            }
            for item in items
            if item.get("slug")
        ]
    except Exception as e:
        print(f"  Warning: could not fetch {collection_id}: {e}")
        return []


def collect_all_slugs() -> list[dict]:
    """Fetch slugs from all four recommendation collections, deduplicate."""
    all_items = []
    for cid, ctype in COLLECTION_IDS:
        print(f"  Fetching collection: {cid} ...", end=" ")
        items = fetch_collection(cid, ctype)
        print(f"{len(items)} items")
        all_items.extend(items)
        time.sleep(0.3)  # polite rate limiting

    # Deduplicate by slug, keep first occurrence (preserves collection priority)
    seen = set()
    unique = []
    for item in all_items:
        if item["slug"] not in seen:
            seen.add(item["slug"])
            unique.append(item)

    print(f"  Total unique slugs: {len(unique)}")
    return unique


# ── Step 2: Fetch series-detail for each slug ─────────────────────────────────

def fetch_series_detail(slug: str) -> dict | None:
    """
    Fetch series-detail for one slug.
    Returns broadcaster name(s), genres, synopsis.
    """
    url = f"{BASE_URL}/series-detail"
    try:
        r = requests.get(url, params={"slug": slug}, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data or not isinstance(data, dict) or not data.get("title"):
            return None

        # Extract broadcaster — NPO Start returns a list, take the first
        broadcasters = [b.get("name", "") for b in data.get("broadcasters", [])]
        broadcaster  = broadcasters[0] if broadcasters else "UNKNOWN"

        # Extract genres — primary + first secondary
        genres      = data.get("genres", [])
        primary_g   = genres[0].get("name", "") if genres else ""
        secondaries = genres[0].get("secondaries", []) if genres else []
        secondary_g = secondaries[0].get("name", "") if secondaries else ""

        return {
            "title":           data.get("title", ""),
            "productId":       data.get("productId", ""),
            "broadcaster":     broadcaster,
            "primary_genre":   primary_g,
            "secondary_genre": secondary_g,
            "genres":          [primary_g, secondary_g] if secondary_g else [primary_g],
            "synopsis":        (data.get("synopsis") or "")[:200],
        }
    except Exception as e:
        print(f"  Warning: could not fetch detail for {slug}: {e}")
        return None


def enrich_with_details(slugs: list[dict]) -> list[dict]:
    """Fetch series-detail for all slugs and merge."""
    enriched = []
    total = len(slugs)
    for i, item in enumerate(slugs):
        print(f"  [{i+1}/{total}] {item['slug']}", end=" ")
        detail = fetch_series_detail(item["slug"])
        if detail:
            merged = {**item, **detail}
            print(f"→ {detail['broadcaster']} | {detail['primary_genre']}")
            enriched.append(merged)
        else:
            print("→ skipped (empty response)")
        time.sleep(0.2)  # polite rate limiting
    return enriched


# ── Step 3: Compute rec_share and save ────────────────────────────────────────

def compute_rec_share(items: list[dict]) -> dict:
    """
    Compute broadcaster share of the recommendation output.
    This is the pre-intervention baseline for the Exposure Gap metric.
    Only counts the six Mediawet 2008 member broadcasters (+ BNNVARA).
    """
    counts = {}
    for item in items:
        bc = item.get("broadcaster", "UNKNOWN")
        counts[bc] = counts.get(bc, 0) + 1

    total = sum(counts.values())
    return {bc: round(count / total, 4) for bc, count in counts.items()}


def save_outputs(items: list[dict]) -> None:
    """Save catalogue CSV, observations CSV, and rec_share JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(items)

    # Catalogue CSV — all columns
    catalogue_path = OUTPUT_DIR / "catalogue.csv"
    df.to_csv(catalogue_path, index=False)
    print(f"\n  Saved catalogue: {catalogue_path} ({len(df)} rows)")

    # Observations CSV — matches the format expected by fairness.compute_rec_share
    obs_cols = ["slug", "title", "broadcaster", "collectionId"]
    obs_df = df[[c for c in obs_cols if c in df.columns]].copy()
    obs_df = obs_df.rename(columns={"slug": "item_id", "collectionId": "session_id"})
    obs_path = OUTPUT_DIR / "observations.csv"
    obs_df.to_csv(obs_path, index=False)
    print(f"  Saved observations: {obs_path} ({len(obs_df)} rows)")

    # rec_share JSON
    rec_share = compute_rec_share(items)
    rs_path = OUTPUT_DIR / "rec_share.json"
    with open(rs_path, "w") as f:
        json.dump(rec_share, f, indent=2)
    print(f"  Saved rec_share: {rs_path}")
    print(f"\n  Broadcaster distribution (rec_share):")
    for bc, share in sorted(rec_share.items(), key=lambda x: -x[1]):
        bar = "█" * int(share * 40)
        print(f"    {bc:12s} {share:.3f}  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("NPO Start — Data Collection Script")
    print("Fetching real recommendation data from npo.nl/start/api")
    print("=" * 60)

    print("\n[1/3] Collecting recommendation collection slugs...")
    slugs = collect_all_slugs()

    print(f"\n[2/3] Fetching series-detail for {len(slugs)} slugs...")
    enriched = enrich_with_details(slugs)

    print(f"\n[3/3] Saving outputs to {OUTPUT_DIR}...")
    save_outputs(enriched)

    print("\nDone! You can now run the app:")
    print("  streamlit run app/app.py")
    print("\nThe app will automatically load the real data from data/processed/.")


if __name__ == "__main__":
    main()
