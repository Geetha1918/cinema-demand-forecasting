# Google Form - answers to copy/paste  (REPLACE the <...> placeholders before submitting)

## Candidate identification
- Full name: **Gorrela Geetha Sri**
- Registration number: **12321952**
- University email: **<your university email>**

## Assignment confirmation
- Internship role: **AI & Machine Learning**
- Exact assigned project title: **Cinema Ticket Demand Forecasting**

## Code and version
- Repository URL: **<https://github.com/YOUR-USERNAME/cinema-demand-forecasting>**
- Primary branch: **main**
- Final commit ID: **<run `git rev-parse HEAD` after your last commit>**
- Reviewer access: public repository (or add reviewer access if private)

## Demonstration
- 3-5 minute video link: **<YouTube unlisted / Google Drive link with "Anyone with the link can view">**
- Live demo URL: **Local-only demo** (run `streamlit run app/streamlit_app.py` or `uvicorn app.api:app`) - or paste your Streamlit Community Cloud URL if you deploy

## Documentation
- Technical report / README link: **<repo URL>/blob/main/reports/TECHNICAL_REPORT.md** and **<repo URL>/blob/main/README.md**
- Architecture/schema diagram: **Pipeline described in README "Layout" and report sections 1-6** (data generator -> clean -> point-in-time features -> chronological split -> baselines + Ridge/RF/HGB -> saved artifact -> FastAPI/Streamlit)

## Technology and limitations
- Technology stack: Python 3.12, pandas, NumPy, scikit-learn (Ridge, RandomForest, HistGradientBoosting), matplotlib, joblib, FastAPI + Uvicorn, Streamlit, pytest.
- Setup notes: `pip install -r requirements.txt`; `python -m src.generate_data`; `python -m src.train`; `pytest`; `uvicorn app.api:app` / `streamlit run app/streamlit_app.py`. Seed fixed at 42.
- Known incomplete features / assumptions: Synthetic data only (generator included), so real-world accuracy is unverified; cold-start 80% intervals under-cover (about 66% on test); demand is censored at capacity so sell-outs are under-forecast; no weather/competitor/promotion features; no API authentication or automated retraining/monitoring.

## Model results (role-specific fields)
- Dataset source/license: Synthetic - generator `src/generate_data.py` (seed 42), 13,685 raw rows, no personal data, free to reuse.
- Notebook/report link: reports/TECHNICAL_REPORT.md in the repository
- Model artifact location: `models/model.joblib` in the repository (also reproducible via `python -m src.train`)
- Evaluation metrics (held-out test, Apr-Jun 2024, 2,274 shows): HistGradientBoosting WAPE 0.194, MAE 15.1 tickets, RMSE 22.6 tickets, staffing-band accuracy 0.811; strongest baseline (weekday x hour capacity-adjusted history) WAPE 0.505, MAE 39.4, RMSE 47.8. Cold-start shows: MAE 23.9 tickets (vs 14.5 for established movies).
- Inference API/app URL: **Local-only:** http://localhost:8000/docs (FastAPI) and http://localhost:8501 (Streamlit)

## AI and attribution
- AI tools used: **Claude (Anthropic)** - used to help design and write the code, tests and documentation, which I then ran, reviewed and can explain. <edit this to describe honestly how YOU used it>
- Purpose: project scaffolding, code drafting, debugging, report drafting.
- Attribution: scikit-learn, pandas, NumPy, matplotlib, FastAPI, Streamlit, pytest (open-source libraries). Dataset is synthetic (no external dataset).

## Declarations (tick only if true)
- Originality: I understand and can explain all code in the submission.
- No real sensitive data or secrets are included (.env.example has placeholders only).
- I have opened every link in a private/incognito window and they all work.
