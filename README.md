# NPO Start · Public Values Recommender System
### Personalisation for (Public) Media · Assignment 2
**Utrecht University | INFOMPPM | 2025–2026**

> **Live demo:** [nporecommender.streamlit.app](https://nporecommender.streamlit.app)

---

## 👥 Group Members and Public Values

| Name | Public Value | Module |
|---|---|---|
| **AmanDeep Singh** | **Fairness** · equitable broadcaster exposure | `src/fairness.py` |
| Padma Dhuney | Diversity · content variety for users | `src/diversity.py` |
| Lisa Wang | Transparency · explanations in the interface | `src/transparency.py` |
| Kiron Putman | Autonomy · user control over recommendations | `src/user_profiles.py` |

> **Primary contributor:** AmanDeep Singh designed and implemented the full system architecture,
> the NPO Start API data pipeline, the Exposure Gap fairness metric, the broadcaster-aware
> re-ranking algorithm, and the Streamlit application. Individual public value modules were
> contributed by each group member within this framework.

---

## 📺 Project Overview

This project builds a working recommender system prototype for **NPO Start**, the on-demand
platform of the Nederlandse Publieke Omroep (NPO). The system addresses a structural fairness
gap in NPO's CTR-optimised pipeline, which systematically underexposes smaller member
broadcasters (VPRO, NTR, EO) in violation of the Mediawet 2008 mandate for balanced
representation.

The prototype integrates four public values into a single Streamlit application:

```
[1] Content-based scoring
    Cosine similarity on genre feature vectors
    + Popularity bias (simulating NPO Start's CTR-driven baseline)
    + User preference boost         <- Autonomy (Kiron Putman)
            |
            v
[2] Diversity re-ranking            <- Diversity (Padma Dhuney)
    Greedy ILS reduction using Jaccard distance on genre tags
            |
            v
[3] Fairness re-ranking             <- Fairness (AmanDeep Singh)
    Broadcaster-aware EG correction with lambda-weighted blended score
    Mediawet 2008 floor: lambda >= 0.10
            |
            v
[4] Explanation labels              <- Transparency (Lisa Wang)
    Human-readable reason on every card + info pop-up
```

---

## 🗂️ Repository Structure

```
publicmedia/
|
+-- app/
|   +-- app.py                  # Streamlit application (main entry point)
|
+-- src/                        # Core algorithm modules
|   +-- data_loader.py          # NPO Start API fetch + broadcaster catalogue counts
|   +-- synthetic_data.py       # Synthetic catalogue and observation data generator
|   +-- scoring.py              # Content-based cosine similarity scoring
|   +-- diversity.py            # ILS diversity re-ranking (Padma Dhuney)
|   +-- fairness.py             # EG fairness re-ranking (AmanDeep Singh)
|   +-- transparency.py         # Explanation label generation (Lisa Wang)
|   +-- user_profiles.py        # Synthetic user generation and autonomy (Kiron Putman)
|
+-- data/
|   +-- processed/              # Pre-fetched real NPO data (committed for Streamlit Cloud)
|       +-- catalogue.csv       # 162 real NPO Start series with broadcaster and genre
|       +-- observations.csv    # Recommendation observation baseline
|       +-- rec_share.json      # Broadcaster recommendation share (pre-intervention)
|       +-- cat_share.json      # Broadcaster catalogue share (from broadcaster pages API)
|
+-- docs/
|   +-- api_walkthrough.md      # NPO Start API endpoint documentation
|   +-- data_loader_guide.md    # How data_loader.py works and what it fetches
|
+-- .devcontainer/              # Streamlit Cloud devcontainer configuration
+-- requirements.txt
+-- .gitignore
+-- README.md
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
source venv/bin/activate        # Mac / Linux
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

The app loads real NPO Start data from `data/processed/` automatically.
If that folder is missing, it falls back to synthetic data.
To re-fetch fresh data from the NPO Start API, run:

```bash
python src/data_loader.py
```

---

## 📊 Data Sources

| Source | Type | Used For |
|---|---|---|
| NPO Start recommendation API | Real · public · no auth | Observation baseline `rec_share` per broadcaster |
| NPO Start broadcaster pages API | Real · public · no auth | Catalogue proportions `cat_share` per broadcaster |
| Synthetic catalogue | Generated · 300 items | Fallback when processed data is unavailable |
| Synthetic users | Generated · 30 profiles | 6 NPO-adapted viewer personas |

The `data/processed/` folder is committed to the repository so that the hosted Streamlit app
at [nporecommender.streamlit.app](https://nporecommender.streamlit.app) loads real data
without requiring a local data collection step.

---

## Algorithm Details

### Fairness Re-ranking (AmanDeep Singh)

Inserts a broadcaster-aware correction at Stage 3 of the pipeline, after content scoring
and diversity re-ranking, before display.

**Exposure Gap metric:**
```
EG = (1/|B|) x sum |rec_share(b) - cat_share(b)|
```

**Re-ranking formula:**
```
final_score(item) = (1 - lambda) x current_score(item)
                  + lambda x fairness_correction_norm(broadcaster(item))

fairness_correction(b)      = max(0, cat_share(b) - rec_share(b))
fairness_correction_norm(b) = fairness_correction(b) / max(fairness_correction)
```

The correction is normalised to [0,1] so it competes on the same scale as the base score.
Lambda is selected via grid search to reduce EG below a target threshold.
A minimum lambda floor of 0.10 is enforced (Mediawet 2008 constraint).

### Diversity Re-ranking (Padma Dhuney)

Greedy selection that penalises genre repetition using Jaccard distance on genre tags.

```
ILS = mean pairwise Jaccard similarity across all item pairs

selection_score(item) = base_score - diversity_factor x mean_similarity_to_selected
```

### Transparency (Lisa Wang)

Every card displays a short reason label. An info pop-up shows feature details and score
breakdown. A general explainer is accessible via the sidebar.

### Autonomy (Kiron Putman)

Lambda slider and genre preference panel in the sidebar. Minimum lambda of 0.10 enforced.

---

## Metrics

| Metric | Baseline | After Re-ranking | Meaning |
|---|---|---|---|
| EG (Exposure Gap) | ~0.045 with real data | Reduced by lambda | Lower = more equitable broadcaster exposure |
| ILS (Intra-List Similarity) | Higher | Lower | Lower = more genre diversity |

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
requests>=2.31.0
plotly>=5.18.0
```
