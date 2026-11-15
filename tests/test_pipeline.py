import numpy as np, pandas as pd, pytest
from src.generate_data import generate
from src.features import clean, build_features, FEATURES, TARGET, LEAD_HOURS
from src import train as T


@pytest.fixture(scope="module")
def small():
    return generate("2023-01-01", "2023-04-30", seed=1)


def test_generator_is_deterministic():
    a = generate("2023-01-01", "2023-01-20", seed=7); b = generate("2023-01-01", "2023-01-20", seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_cleaning_fixes_quality_issues(small):
    df, rep = clean(small)
    assert df.show_id.is_unique and df.tickets_sold.min() >= 0
    assert (df.tickets_sold <= df.screen_capacity).all() and df.ticket_price.notna().all()
    assert df[TARGET].between(0, 1).all()


def test_no_target_leakage_in_history_features(small):
    """Changing the target of the show itself AND of every later show must not change its features."""
    df, _ = clean(small)
    X1 = build_features(df, df)
    df2 = df.copy()
    cut = df2.show_datetime.quantile(0.6)
    df2.loc[df2.show_datetime >= cut - pd.Timedelta(hours=LEAD_HOURS - 1), TARGET] = 0.123   # corrupt "future"
    X2 = build_features(df, df2)
    early = X1.show_datetime < cut          # these shows' 48h-cutoff precedes the corrupted region
    pd.testing.assert_frame_equal(X1.loc[early, FEATURES].reset_index(drop=True),
                                  X2.loc[early, FEATURES].reset_index(drop=True))


def test_history_uses_only_shows_older_than_lead_time(small):
    df, _ = clean(small)
    X = build_features(df, df)
    row = X[X.movie_prior_shows > 5].iloc[0]
    cutoff = row.show_datetime - pd.Timedelta(hours=LEAD_HOURS)
    past = df[(df.movie_id == row.movie_id) & (df.show_datetime <= cutoff)]
    assert row.movie_prior_shows == len(past)
    assert np.isclose(row.movie_hist_mean_occ, past[TARGET].mean())


def test_features_do_not_contain_target_or_ids():
    banned = {"tickets_sold", "occupancy", "show_id", "movie_id", "show_datetime"}
    assert not banned & set(FEATURES)


def test_metrics_wape():
    m = T.metrics(np.array([0.5, 1.0]), np.array([0.5, 0.5]), np.array([100, 100]))
    assert np.isclose(m["WAPE"], 50 / 150) and np.isclose(m["MAE_tickets"], 25)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    data = d / "shows.csv"
    generate("2023-01-01", "2023-08-31", seed=3).to_csv(data, index=False)
    out = T.main(["--data", str(data), "--model-out", str(d / "m.joblib"), "--reports", str(d / "rep"),
                  "--fast", "--train-end", "2023-05-31", "--val-end", "2023-06-30"])
    return d, out


def test_chronological_split(trained):
    _, out = trained
    s = out["split"]
    assert s["train"][1] < s["validation"][0] <= s["validation"][1] < s["test"][0]


def test_model_beats_naive_baseline(trained):
    _, out = trained
    r = out["metrics"]["test"]
    assert r[out["best_model"]]["WAPE"] < r["naive_global_mean"]["WAPE"]


def test_inference_loads_saved_artifact_and_is_bounded(trained):
    from src.predict import Forecaster
    d, _ = trained
    f = Forecaster(str(d / "m.joblib"), str(d / "shows.csv"))
    rec = dict(movie_id="NEW", genre="Action", language="Hindi", runtime_min=130, is_franchise=1,
               pre_release_buzz=70, release_date="2023-08-25", screen_capacity=150,
               show_datetime="2023-08-26T19:00:00", ticket_price=200)
    r = f.predict(rec)
    assert 0 <= r["predicted_tickets"] <= 150 and r["data_availability_group"].startswith("cold")
    lo, hi = r["interval_80pct_tickets"]; assert 0 <= lo <= r["predicted_tickets"] <= hi <= 150
    with pytest.raises(ValueError):
        f.predict({**rec, "genre": None})
    r2 = f.predict({**rec, "pre_release_buzz": 10, "genre": "Drama"})
    assert r2["predicted_tickets"] != r["predicted_tickets"]      # model actually responds to inputs
