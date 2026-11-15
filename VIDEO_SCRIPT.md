# 3-5 minute demo video script (screen recording)

**0:00-0:30 Intro.** "I'm Gorrela Geetha Sri, reg. 12321952. My project forecasts ticket sales per show, 48 hours before it starts, so a cinema can allocate screens and staff. It is decision support - a human reviews every forecast."

**0:30-1:30 Data + EDA.** Open `reports/figures/eda.png`. Say: synthetic data, 13,685 shows, 5 screens, 18 months; occupancy rises on weekends and at 19:00, decays after release; sales are capped at capacity (censoring); I cleaned duplicates, negative counts, over-capacity values and missing prices.

**1:30-2:15 Leakage + split.** Show `src/features.py` (the `lead_hours` logic) and run `pytest -q` (11 pass, including the leakage test). Explain the chronological split: train 2023, validation Q1 2024, test Q2 2024; tuning on validation only.

**2:15-3:15 Results.** Show the test table in `reports/TECHNICAL_REPORT.md`: baseline WAPE 0.505 / MAE 39.4 vs HGB WAPE 0.194 / MAE 15.1. Show the by-group table: cold-start MAE 23.9 vs established 14.5. Show `wape_by_group.png`.

**3:15-4:30 Live prediction.** Run `streamlit run app/streamlit_app.py` (or `/docs` in FastAPI). Predict 1) an existing movie and 2) a brand-new movie (cold start). Point out the tickets, the 80% interval, the warning, and the "why this forecast" bars. Say these come from `models/model.joblib`, not hard-coded values.

**4:30-5:00 Limitations.** Synthetic data; cold-start intervals too narrow; sell-outs under-forecast by ~17 tickets because demand is censored; no weather/competitor features; needs real data and human review.
