# data_loader.py · Guide
**Author: AmanDeep Singh**
**Purpose: Explains how data_loader.py works, what it fetches, and where the outputs go**

---

## What it does

`src/data_loader.py` is a standalone script that fetches real data from the NPO Start public API and saves it to `data/processed/`. The app loads these files automatically. If they are missing, it falls back to synthetic data.

Run it once locally before deploying:
```bash
python src/data_loader.py
```

---

## Four-step process

### Step 1 — Collect slugs from recommendation collections
Calls 11 recommendation rows from NPO Start's homepage (the content shown to anonymous users). Each row represents a different editorial or algorithmic curation. Deduplicates by slug to get ~162-250 unique series.

### Step 2 — Enrich with series-detail
For each unique slug, calls the `series-detail` endpoint to get broadcaster name, genre tags, synopsis, and image URLs. Skips items with empty responses.

### Step 3 — Save observation outputs
Writes three files to `data/processed/`:
- `catalogue.csv` — full series metadata including broadcaster, genres, images
- `observations.csv` — stripped-down version used as rec_share baseline
- `rec_share.json` — broadcaster share of the recommendation output (pre-intervention baseline)

### Step 4 — Fetch catalogue shares from broadcaster pages
For each broadcaster (avrotros, bnnvara, kro-ncrv, max, ntr, eo, vpro), calls their page-layout endpoint to get collection GUIDs, then calls page-collection for each GUID to count unique items. Computes proportions and saves:
- `cat_share.json` — empirically-derived broadcaster catalogue share

---

## Output files

| File | Content | Used by |
|---|---|---|
| `catalogue.csv` | ~162 real NPO Start series | `app.py` main data source |
| `observations.csv` | Recommendation observation baseline | EG computation |
| `rec_share.json` | Broadcaster share of recommendations | `fairness.py` baseline |
| `cat_share.json` | Broadcaster share of full catalogue | `fairness.py` cat_share |

---

## Design decisions

**Why commit the outputs to GitHub?**
Streamlit Cloud cannot reach NPO Start's API directly. Committing `data/processed/` means the hosted app loads real data without requiring a local run.

**Why 11 collection rows?**
The original 4 rows yielded only ~89 unique series, which was too small for a reliable `rec_share` baseline. 11 rows gives ~162-250 items, which is more representative.

**Why use rec_share from observations rather than computing it from the catalogue?**
`rec_share` should reflect what the current CTR-optimised system actually recommends, not what is available. Computing it from observations measures the system's actual output, which is the pre-intervention baseline for the EG fairness metric.

**Why is cat_share fetched separately?**
Computing `cat_share` from the recommendation sample would produce EG = 0 by construction (the sample IS the recommendation). The catalogue share must come from the full broadcaster portfolio, which is fetched from their individual NPO Start pages.

---

## Fallback behaviour

If `catalogue.csv` does not exist, the app generates synthetic data using `src/synthetic_data.py`. The synthetic catalogue has 300 items with broadcaster proportions grounded in real NPO portfolio sizes. EG values will differ from the real-data case but the intervention still demonstrates correctly.

The app shows 🟢 Real data or 🟡 Synthetic data in the top navigation bar.
