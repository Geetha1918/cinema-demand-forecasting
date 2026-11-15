"""Inference: loads the SAVED model artifact and the observed-show history table, builds
point-in-time features for a requested show, and returns forecast + interval + explanation."""
import os
import numpy as np
import pandas as pd
import joblib
from .features import clean, build_features

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
HISTORY_PATH = os.environ.get("HISTORY_PATH", "data/shows.csv")

REQUIRED = ["movie_id", "genre", "language", "runtime_min", "is_franchise", "pre_release_buzz",
            "release_date", "screen_capacity", "show_datetime", "ticket_price"]
GROUPS = {  # feature groups for the local "what-if" explanation
    "weekday / hour / month": ["weekday", "hour", "month", "is_weekend"],
    "holiday": ["is_holiday"],
    "days since release": ["days_since_release"],
    "this movie's sales history": ["movie_prior_shows", "movie_hist_mean_occ",
                                   "movie_recent7_mean_occ", "movie_recent7_count"],
    "genre trend": ["genre_hist_mean_occ"],
    "weekday-hour slot history": ["slot_hist_mean_occ"],
    "pre-release buzz": ["pre_release_buzz"],
    "ticket price": ["ticket_price"],
    "genre / language / franchise / runtime": ["genre", "language", "is_franchise", "runtime_min"],
}


class Forecaster:
    def __init__(self, model_path=MODEL_PATH, history_path=HISTORY_PATH):
        self.art = joblib.load(model_path)
        self.history, _ = clean(pd.read_csv(history_path))
        self.last_history = self.history["show_datetime"].max()

    def _features(self, rec):
        miss = [k for k in REQUIRED if rec.get(k) is None]
        if miss:
            raise ValueError(f"missing fields: {miss}")
        df = pd.DataFrame([rec])
        df["show_datetime"] = pd.to_datetime(df["show_datetime"])
        df["release_date"] = pd.to_datetime(df["release_date"])
        if int(df.screen_capacity.iloc[0]) <= 0:
            raise ValueError("screen_capacity must be positive")
        return build_features(df, self.history)

    def predict(self, rec):
        X = self._features(rec)
        art, feats = self.art, self.art["features"]
        cap = float(X.screen_capacity.iloc[0])
        occ = float(np.clip(art["pipeline"].predict(X[feats])[0], 0, 1))
        n_prior = int(X.movie_prior_shows.iloc[0])
        group = ("cold_start(0 prior shows)" if n_prior == 0 else
                 "low_data(1-9 prior shows)" if n_prior < 10 else "established(>=10 prior shows)")
        lo_r, hi_r = art["residual_q"][group]
        lo = float(min(np.clip(occ + lo_r, 0, 1), occ))   # interval always contains the point forecast
        hi = float(max(np.clip(occ + hi_r, 0, 1), occ))
        # local what-if explanation: replace each feature group by a reference value
        contrib = []
        for name, cols in GROUPS.items():
            Z = X.copy()
            for c in cols:
                Z[c] = art["ref"][c]
            ref_occ = float(np.clip(art["pipeline"].predict(Z[feats])[0], 0, 1))
            contrib.append({"factor": name, "effect_on_tickets": round((occ - ref_occ) * cap, 1)})
        contrib.sort(key=lambda d: -abs(d["effect_on_tickets"]))
        warns = []
        if group != "established(>=10 prior shows)":
            warns.append(f"Limited sales history for this movie ({n_prior} prior shows known 48h before "
                         "the show): forecast relies mostly on buzz, genre and calendar; expect larger error.")
        if occ >= 0.9:
            warns.append("Forecast is near capacity; true demand may be higher (sales are capped by seats).")
        stale = (X.show_datetime.iloc[0] - pd.Timedelta(hours=art["lead_hours"]) - self.last_history).days
        if stale > 7:
            warns.append(f"History table ends {stale} days before this forecast's cut-off; "
                         "history features are stale - retrain/refresh data.")
        return {"predicted_tickets": round(occ * cap, 1), "predicted_occupancy": round(occ, 3),
                "interval_80pct_tickets": [round(lo * cap, 1), round(hi * cap, 1)],
                "data_availability_group": group, "movie_prior_shows_known": n_prior,
                "model": art["best_name"], "top_factors": contrib[:5], "warnings": warns,
                "note": "Decision-support only: a human scheduler should review before allocating screens/staff."}
