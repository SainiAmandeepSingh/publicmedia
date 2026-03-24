# app/app.py
# NPO Start — Public Values Recommender System
# Streamlit application entry point
#
# Run with: streamlit run app/app.py

# TODO: Build after group confirms pipeline architecture decisions
# Components to integrate:
# - src/data_loader.py     (data)
# - src/scoring.py         (base scores)
# - src/diversity.py       (Padma — ILS re-ranking)
# - src/fairness.py        (AmanDeep — EG re-ranking)
# - src/transparency.py    (Lisa — explanation labels)
# - src/user_profiles.py   (Kiron — user controls)

import streamlit as st

st.set_page_config(
    page_title="NPO Start — Fairness Recommender",
    page_icon="📺",
    layout="wide",
)

st.title("📺 NPO Start — Public Values Recommender")
st.info("App coming soon. Run notebooks/01_data_collection.ipynb first.")
