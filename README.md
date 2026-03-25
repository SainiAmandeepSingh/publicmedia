# NPO Start — Public Values Recommender System
### Personalisation for (Public) Media — Assignment 2
**Utrecht University | INFOMPPM | 2025–2026**

---

## 👥 Group Members & Public Values

| Name | Public Value | Module |
|---|---|---|
| AmanDeep Singh | **Fairness** — equitable broadcaster exposure | `src/fairness.py` |
| Padma Dhuney | **Diversity** — content variety for users | `src/diversity.py` |
| Lisa Wang | **Transparency** — explanations in the interface | `src/transparency.py` |
| Kiron Putman | **Autonomy** — user control over recommendations | `src/user_profiles.py` |

---

## 📺 Project Overview

This project builds a working recommender system prototype for **NPO Start**, the on-demand platform of the Nederlandse Publieke Omroep (NPO). The system addresses a structural fairness gap in NPO's CTR-optimised pipeline, which systematically underexposes smaller member broadcasters (VPRO, NTR, EO) in violation of the Mediawet 2008 mandate for balanced representation.

The prototype integrates four public values into a single Streamlit application:

```
[1] Content-based scoring
    Cosine similarity on genre feature vectors
    + Popularity bias (simulating NPO Start's CTR-driven baseline)
    + User preference boost  ← Autonomy (Kiron Putman)
            ↓
[2] Diversity re-ranking     ← Diversity (Padma Dhuney)
    Greedy ILS reduction using Jaccard distance on genre tags
            ↓
[3] Fairness re-ranking      ← Fairness (AmanDeep Singh)
    Broadcaster-aware EG correction with λ-weighted blended score
    Mediawet 2008 floor: λ ≥ 0.10
            ↓
[4] Explanation labels       ← Transparency (Lisa Wang)
    Human-readable reason on every card + info pop-up
```

---

## 🗂️ Repository Structure

```
publicmedia/
│
├── app/
│   └── app.py                  # Streamlit application — main entry point
│
├── src/                        # Core algorithm modules
│   ├── data_loader.py          # NPO POMS API fetch + fallback logic
│   ├── synthetic_data.py       # Synthetic catalogue + observation data generator
│   ├── scoring.py              # Content-based cosine similarity scoring
│   ├── diversity.py            # ILS diversity re-ranking (Padma Dhuney)
│   ├── fairness.py             # EG fairness re-ranking (AmanDeep Singh)
│   ├── transparency.py         # Explanation label generation (Lisa Wang)
│   └── user_profiles.py        # Synthetic user generation + autonomy (Kiron Putman)
│
├── data/
│   ├── raw/                    # Raw data from POMS API or observation sampling
│   └── processed/              # Cleaned datasets ready for use
│
├── notebooks/
│   └── 01_data_collection.ipynb   # Optional: POMS API pull + observation sampling
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/SainiAmandeepSingh/publicmedia.git
cd publicmedia
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app/app.py
```

The app runs on synthetic data by default — no data collection step required.
If you want to try pulling real catalogue data from the POMS API, see `src/data_loader.py`.

---

## 📊 Data Sources

| Source | Type | Used For |
|---|---|---|
| [NPO POMS API](https://api.poms.omroep.nl) | Real — public, no auth required | Content catalogue: broadcaster labels, genre tags, episode metadata |
| NPO Start "Aanbevolen voor jou" | Real — observation sampling | Baseline `rec_share` per broadcaster (pre-intervention) |
| Synthetic catalogue | Generated — 300 items | Fallback when POMS API is unavailable; grounded in real broadcaster portfolio sizes |
| Synthetic users | Generated — 30 profiles | 6 NPO-adapted viewer personas with realistic genre distributions |

**Note on POMS API access:** The API is publicly accessible from a local machine via standard HTTP requests without authentication (`api.poms.omroep.nl`). It is not reachable from sandboxed cloud environments. Run `src/data_loader.py` locally to fetch and save real catalogue data to `data/processed/catalogue.csv`, which the app will load automatically on subsequent runs.

---

## ⚙️ Algorithm Details

### Fairness Re-ranking (AmanDeep Singh)
Inserts a broadcaster-aware correction at Stage 3 of the pipeline — after content scoring and diversity re-ranking, before display.

**Exposure Gap metric:**
```
EG = (1/|B|) × Σ |rec_share(b) − cat_share(b)|
```

**Re-ranking formula:**
```
final_score(item) = (1−λ) × current_score(item) + λ × fairness_correction_norm(broadcaster(item))

fairness_correction(b) = max(0, cat_share(b) − rec_share(b))
fairness_correction_norm = fairness_correction / max(fairness_correction)
```

The correction is normalised to [0,1] so it is on the same scale as the base score. λ is selected via grid search on the observation sample to reduce EG below a target threshold while preserving acceptable engagement performance. A **minimum λ floor of 0.10** is enforced — this represents the Mediawet 2008 constraint that broadcaster fairness cannot be fully disabled by user preferences.

### Diversity Re-ranking (Padma Dhuney)
Greedy selection algorithm that penalises genre repetition using Jaccard distance on genre tags.

```
ILS = mean pairwise Jaccard similarity across all item pairs in the list

selection_score(item) = base_score(item) − diversity_factor × mean_similarity_to_already_selected
```

### Transparency (Lisa Wang)
Every recommended item displays:
- A short reason label on the card ("📺 Matches your interest in Drama", "🟢 Supporting VPRO — smaller broadcaster gaining visibility")
- An ℹ️ pop-up with feature details and score breakdown
- A ❓ general algorithm explainer accessible from the profile tab

### Autonomy (Kiron Putman)
User-controlled settings exposed in the sidebar:
- **λ slider** — fairness weight, bounded by Mediawet 2008 floor (min 0.10)
- **Genre preference panel** — explicit genre interests that boost base scores
- **Profile page** — view watch history, current settings, reset preferences

---

## ⚖️ Key Design Decision: Autonomy vs. Fairness Floor

Kiron's autonomy features give users direct control over λ. However, NPO operates under the **Mediawet 2008**, which mandates balanced representation across member broadcasters. The system enforces a **minimum λ floor** to ensure the Exposure Gap never exceeds an acceptable threshold even when users maximise personal relevance.

This is a principled compromise: user autonomy operates *within* NPO's legal mandate, not over it. The trade-off is documented in the report's critical reflection section.

---

## 📏 Metrics Summary

| Metric | Baseline | After re-ranking | Meaning |
|---|---|---|---|
| EG (Exposure Gap) | ~0.09–0.14 | Reduced by λ | Lower = more equitable broadcaster exposure |
| ILS (Intra-List Similarity) | Higher | Lower | Lower = more genre diversity in the list |

*EG baseline depends on data source: ~0.09 with synthetic data, ~0.14 with real POMS API + observation sampling as measured in AmanDeep Singh's research proposal.*

---

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
requests>=2.31.0
plotly>=5.18.0
matplotlib>=3.8.0
jupyter>=1.0.0
```
