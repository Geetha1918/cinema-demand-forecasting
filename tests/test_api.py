import os
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not os.path.exists("models/model.joblib"),
                                reason="run `python -m src.train` first")
SAMPLE = dict(movie_id="M126", genre="Comedy", language="Hindi", runtime_min=127, is_franchise=0,
              pre_release_buzz=44.9, release_date="2024-06-28", screen_capacity=90,
              show_datetime="2024-06-29T19:00:00", ticket_price=192)


def test_health_and_predict():
    from app.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json=SAMPLE)
    assert r.status_code == 200 and 0 <= r.json()["predicted_tickets"] <= 90


def test_validation_error():
    from app.api import app
    assert TestClient(app).post("/predict", json={**SAMPLE, "screen_capacity": -5}).status_code == 422
