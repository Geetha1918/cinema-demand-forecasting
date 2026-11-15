# Cinema Ticket Demand Forecasting

**Candidate:** Gorrela Geetha Sri | **Reg. No.:** 12321952 | **Role:** AI & Machine Learning - Algions AI Labs

Show-level forecast of **tickets / seat occupancy**, made **48 hours before the show**, to help allocate screens and staff.
Decision-support only: a human reviews every forecast. Full details: [`reports/TECHNICAL_REPORT.md`](reports/TECHNICAL_REPORT.md).

## Layout
```
src/generate_data.py   seeded synthetic data generator (no real/personal data)
src/features.py        cleaning + leakage-safe point-in-time features
src/train.py           chronological split, baselines, 3 models, metrics, artifact, figures
src/predict.py         loads saved model -> forecast + interval + explanation
src/make_report.py     fills report tables from reports/metrics.json
app/api.py             FastAPI  (POST /predict, GET /health)
app/streamlit_app.py   interactive UI
tests/                 leakage, split, cleaning, metric, inference, API tests
models/model.joblib    saved artifact      reports/  metrics, figures, importance
```

## Setup (Python 3.10+)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce data, split and training (seed = 42)
```bash
python -m src.generate_data        # -> data/shows.csv (13,685 raw rows)
python -m src.train                # -> models/model.joblib, reports/metrics.json, figures
python -m src.make_report          # -> reports/TECHNICAL_REPORT.md
```
Split (chronological): train 2023-01-01..2023-12-31, validation 2024-01-01..2024-03-31, test 2024-04-01..2024-06-30.
Hyper-parameters are tuned on validation only; the test set is evaluated once. `python -m src.train --fast` runs a quick smoke version.

## Tests
```bash
pytest -q          # 11 tests (leakage, split, cleaning, metrics, inference, API)
```

## Inference
**API**
```bash
uvicorn app.api:app --port 8000
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{
 "movie_id":"NEW","genre":"Action","language":"Hindi","runtime_min":140,"is_franchise":1,
 "pre_release_buzz":80,"release_date":"2024-06-28","screen_capacity":240,
 "show_datetime":"2024-06-29T19:00:00","ticket_price":220}'
```
Interactive docs: http://localhost:8000/docs. **App:** `streamlit run app/streamlit_app.py`.
The response contains predicted tickets/occupancy, an 80% interval, the data-availability group (cold start / low data / established), the top factors and warnings. Predictions come from the saved model (`models/model.joblib`); history features are computed from `data/shows.csv`.

## Known limitations
Synthetic data; cold-start intervals under-cover; no weather/competitor features; demand censored at capacity; local demo without authentication. See the report, section 10-11.
