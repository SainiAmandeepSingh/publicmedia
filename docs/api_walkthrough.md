# NPO Start API · Walkthrough
**Author: AmanDeep Singh**
**Purpose: Documents the NPO Start public API endpoints used to collect real data for the fairness analysis**

---

## Overview

NPO Start exposes a public JSON API at `https://npo.nl/start/api/domain`. No authentication is required. All endpoints are accessible via standard HTTP GET requests from a local machine.

This document describes the four endpoints used in `src/data_loader.py` to fetch real recommendation data and broadcaster catalogue proportions for the Exposure Gap (EG) fairness metric.

---

## Endpoint 1: recommendation-collection

**Purpose:** Fetches one row of content recommendations as shown on NPO Start's homepage.

**URL:**
```
GET https://npo.nl/start/api/domain/recommendation-collection
```

**Parameters:**
| Parameter | Value | Description |
|---|---|---|
| `collectionId` | e.g. `series-anonymous-v0` | Identifies which recommendation row to fetch |
| `collectionType` | `SERIES` or `PROGRAM` | Content type filter |
| `includePremiumContent` | `true` | Include premium items |
| `layoutType` | `RECOMMENDATION` | Request type |
| `partyId` | any session token | Anonymous session identifier |

**Collection IDs used (11 rows):**
- `series-anonymous-v0` — main recommendations row
- `trending-anonymous-v0` — trending series
- `public-value-anonymous-v0` — public value curated row
- `recent-free-v0` — recently added free content
- `real-life-anonymous-v0` — real-life / human interest
- `crime-anonymous-v0` — crime and thriller
- `documentaries-anonymous-v0` — documentary programmes
- `documentaries-series-v0` — documentary series
- `films-anonymous-v0` — films
- `youth-0-6-v0` — children 0-6
- `youth-6-12-v0` — children 6-12

**Returns:** JSON with `items` array, each item containing `slug`, `title`, `productId`, `itemRecommender`.

---

## Endpoint 2: series-detail

**Purpose:** Fetches full metadata for a single series by slug. Used to get broadcaster name, genre tags, and image URLs.

**URL:**
```
GET https://npo.nl/start/api/domain/series-detail?slug=<slug>
```

**Returns:** JSON with `title`, `broadcasters` (list), `genres` (list with `name` and `secondaries`), `images` (list with `role` and `url`), `contentClassification`.

**Key fields extracted:**
- `broadcasters[0].name` → broadcaster label (e.g. `VPRO`, `EO`, `AVROTROS`)
- `genres[0].name` → primary genre
- `genres[0].secondaries[0].name` → secondary genre
- `images` filtered by `role=default` → card image URL

---

## Endpoint 3: page-layout

**Purpose:** Fetches a broadcaster's page structure on NPO Start. Returns a list of collection GUIDs that make up the broadcaster's catalogue page.

**URL:**
```
GET https://npo.nl/start/api/domain/page-layout?layoutId=<broadcaster-slug>&layoutType=PAGE
```

**Broadcaster slugs used:** `avrotros`, `bnnvara`, `kro-ncrv`, `max`, `ntr`, `eo`, `vpro`

**Returns:** JSON with `title` (broadcaster display name) and `collections` (list of `{collectionId, type}` objects).

---

## Endpoint 4: page-collection

**Purpose:** Fetches all items in a broadcaster's catalogue collection. Used together with `page-layout` to count unique programmes per broadcaster.

**URL:**
```
GET https://npo.nl/start/api/domain/page-collection
```

**Parameters:** Same as recommendation-collection but with `layoutType=PAGE`.

**Returns:** JSON with `items` array, each with a `slug`. Unique slugs are counted to derive `cat_share(b)`.

---

## Data collection flow

```
1. Call recommendation-collection (11 rows) → collect slugs
2. Call series-detail for each unique slug → broadcaster + genre per item
3. Save as catalogue.csv + observations.csv + rec_share.json
4. Call page-layout for each broadcaster → collection GUIDs
5. Call page-collection for each GUID → count unique items
6. Compute proportions → save as cat_share.json
```

Full implementation: `src/data_loader.py`

---

## Notes

- Rate limiting: 0.2s delay between series-detail calls, 0.3s between collections
- Sandbox environments (Streamlit Cloud) cannot reach NPO Start's API. Run `data_loader.py` locally and commit the outputs to `data/processed/`.
- The `partyId` parameter accepts any string. A fresh one is generated each session.
- Fetching all 11 collection rows yields ~162-250 unique series, giving a more reliable `rec_share` baseline than the original 4 rows (~89 items).
