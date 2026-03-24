# NPO Start — Public Values Recommender System
### Personalisation for (Public) Media — Assignment 2
**Utrecht University | INFOMPPM | 2025–2026**

---

## 👥 Group Members & Public Values

| Name | Public Value | Role |
|---|---|---|
| AmanDeep Singh | **Fairness** — equitable broadcaster exposure | Fairness re-ranking algorithm (EG metric) |
| Padma Dhuney | **Diversity** — content variety for users | Diversity re-ranking (ILS metric) |
| Lisa Wang | **Transparency** — explanations in the interface | Explanation labels & UI |
| Kiron Putman | **Autonomy** — user control over recommendations | User preference controls & profile settings |

---

## 📺 Project Overview

This project builds a working recommender system prototype for **NPO Start**, the on-demand platform of the Nederlandse Publieke Omroep (NPO). The system addresses a structural fairness gap in NPO's CTR-optimised pipeline, which systematically underexposes smaller member broadcasters (VPRO, NTR, EO) in violation of the Mediawet 2008 mandate for balanced representation.

The prototype integrates four public values into a single pipeline:

```
User preference settings (Autonomy)
        ↓
Content-based scoring (watch history + genre match)
        ↓
Diversity re-ranking — ILS correction (Diversity)
        ↓
Fairness re-ranking — EG correction (Fairness)
        ↓
Display with explanation labels on every card (Transparency)
```

---

## 🗂️ Repository Structure

```
npo_recommender/
│
├── app/                        # Streamlit application
│   └── app.py                  # Main app entry point
│
├── src/                        # Core algorithm modules
│   ├── data_loader.py          # POMS API + data loading (MAYBE)
│   ├── scoring.py              # Base CTR / cosine similarity scoring (MAYBE)
│   ├── diversity.py            # ILS diversity re-ranking (Padma)
│   ├── fairness.py             # EG fairness re-ranking (AmanDeep)
│   ├── transparency.py         # Explanation label generation (Lisa)
│   └── user_profiles.py        # Synthetic user generation (Kiron)
│
├── data/
│   ├── raw/                    # Raw data from POMS API + observation sampling
│   └── processed/              # Cleaned, merged datasets ready for use
│
├── notebooks/
│   ├── 01_data_collection.ipynb        # POMS API pull + observation sampling (MAYBE)

├── docs/
│   └── report.md               # Draft report sections
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Files to exclude from git
└── README.md                   # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/npo_recommender.git
cd npo_recommender
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

### 4. Collect data
Run the data collection notebook first:
```bash
jupyter notebook notebooks/01_data_collection.ipynb
```

### 5. Run the Streamlit app
```bash
streamlit run app/app.py
```

---

## 📊 Data Sources

| Source | Type | Used For |
|---|---|---|
| [NPO POMS API](https://api.poms.omroep.nl) | Real — public API | Content catalogue (broadcaster, genre, metadata) |
| NPO Start "Aanbevolen voor jou" | Real — observation sampling | Baseline rec_share per broadcaster |
| Synthetic users | Generated — based on real genre distribution | User profiles, watch history, preferences |

---

## ⚙️ Algorithm Summary

### Fairness Re-ranking (AmanDeep)
Corrects broadcaster exposure at Stage 3 of the pipeline.

```
EG = (1/|B|) × Σ |rec_share(b) − cat_share(b)|

final_score(item) = (1−λ) × current_score(item) + λ × fairness_correction(broadcaster(item))

fairness_correction(b) = max(0, cat_share(b) − rec_share(b))
```

λ is selected via grid search to minimise EG while preserving acceptable engagement performance.

### Diversity Re-ranking (Padma)
Reduces Intra-List Similarity (ILS) using Jaccard distance on genre tags.

```
ILS = mean(jaccard_similarity(item_i, item_j)) for all pairs in top-N

diversity_score(item) = 1 − mean_similarity_to_already_selected_items
```

### Transparency (Lisa)
Every recommended item displays:
- Primary reason label ("Because you watched...", "Popular in Documentary", "Fairness-boosted")
- ℹ️ button → pop-up with features that drove the recommendation
- ❓ button → explanation of what data the algorithm uses overall

### Autonomy (Kiron)
User-controlled settings in the sidebar:
- **λ slider** — fairness weight (with minimum floor enforced by Mediawet 2008 constraint)
- **Genre preference panel** — explicit genre interests that override behavioural data
- **Profile transparency page** — view and edit what the system thinks you like

---

## 📏 Metrics

| Metric | Value | Meaning |
|---|---|---|
| EG (before) | ~0.14 | Baseline exposure gap |
| EG (after) | ~0.02 | Post-intervention exposure gap |
| ILS (before) | High | Low genre diversity |
| ILS (after) | Lower | Higher genre diversity |

---

## ⚖️ Key Design Decision: Autonomy vs. Fairness Floor

Kiron's autonomy features allow users to set λ themselves. However, NPO operates under the **Mediawet 2008**, which mandates balanced broadcaster representation. To respect both values, the system enforces a **minimum fairness floor**: λ cannot be set below a minimum threshold that ensures EG stays below a maximum acceptable level. Users have control within this floor, not over it. This design choice is discussed in the report's critical reflection.

---
