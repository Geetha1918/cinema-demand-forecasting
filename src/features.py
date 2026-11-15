"""Cleaning + leakage-safe feature engineering.

PREDICTION POINT: a forecast for a show is made LEAD_HOURS (=48h) before show start.
Every history feature for a show at time t uses ONLY shows that started at or before
(t - 48h), i.e. shows whose final ticket counts are already known at prediction time.
The show's own target and any later show can never enter its features.
"""
import numpy as np
import pandas as pd
from .generate_data import HOLIDAYS

LEAD_HOURS = 48
CAT = ["genre", "language"]
NUM = ["screen_capacity", "ticket_price", "runtime_min", "is_franchise", "pre_release_buzz",
       "weekday", "hour", "month", "is_weekend", "is_holiday", "days_since_release",
       "movie_prior_shows", "movie_hist_mean_occ", "movie_recent7_mean_occ",
       "movie_recent7_count", "genre_hist_mean_occ", "slot_hist_mean_occ"]
FEATURES = CAT + NUM
TARGET = "occupancy"


def clean(df):
    """Data-quality handling. Returns (clean_df, report_dict)."""
    rep = {"rows_raw": len(df)}
    df = df.copy()
    df["show_datetime"] = pd.to_datetime(df["show_datetime"])
    df["release_date"] = pd.to_datetime(df["release_date"])
    df = df.drop_duplicates(subset=["show_id"])
    rep["duplicates_removed"] = rep["rows_raw"] - len(df)
    bad = df["tickets_sold"] < 0
    rep["negative_tickets_dropped"] = int(bad.sum())
    df = df[~bad]
    over = df["tickets_sold"] > df["screen_capacity"]
    rep["over_capacity_clipped"] = int(over.sum())
    df.loc[over, "tickets_sold"] = df.loc[over, "screen_capacity"]
    rep["missing_price_imputed"] = int(df["ticket_price"].isna().sum())
    df["ticket_price"] = df["ticket_price"].fillna(
        df.groupby(["screen_id", df["show_datetime"].dt.hour])["ticket_price"].transform("median"))
    df[TARGET] = df["tickets_sold"] / df["screen_capacity"]
    df = df.sort_values("show_datetime").reset_index(drop=True)
    rep["rows_clean"] = len(df)
    return df, rep


def add_calendar(df):
    d = df.copy()
    dt = d["show_datetime"]
    d["weekday"] = dt.dt.weekday
    d["hour"] = dt.dt.hour
    d["month"] = dt.dt.month
    d["is_weekend"] = (d["weekday"] >= 4).astype(int)  # Fri-Sun
    d["is_holiday"] = dt.dt.normalize().isin(HOLIDAYS).astype(int)
    d["days_since_release"] = (dt.dt.normalize() - d["release_date"]).dt.days.clip(lower=0)
    return d


def _key(df, keys):
    return df[keys].astype(str).agg("|".join, axis=1)


def _hist_stats(query, hist, keys, prefix, lead_hours=LEAD_HOURS, window_days=7):
    """For each query row: count / mean / 7-day mean occupancy of `hist` rows with the same
    key whose start time <= query_time - lead_hours."""
    n = len(query)
    cnt, mean = np.zeros(n), np.full(n, np.nan)
    rcnt, rmean = np.zeros(n), np.full(n, np.nan)
    hk, qk = _key(hist, keys), _key(query, keys)
    hgroups = {k: g for k, g in hist.assign(_k=hk).sort_values("show_datetime").groupby("_k")}
    qidx = pd.Series(np.arange(n)).groupby(qk.values).indices
    qt = query["show_datetime"].values
    for k, idx in qidx.items():
        g = hgroups.get(k)
        if g is None:
            continue
        ht = g["show_datetime"].values
        cum = np.concatenate([[0.0], np.cumsum(g[TARGET].values)])
        tp = qt[idx] - np.timedelta64(lead_hours, "h")
        hi = np.searchsorted(ht, tp, side="right")
        lo = np.searchsorted(ht, tp - np.timedelta64(window_days, "D"), side="right")
        cnt[idx] = hi
        with np.errstate(invalid="ignore", divide="ignore"):
            mean[idx] = np.where(hi > 0, cum[hi] / np.maximum(hi, 1), np.nan)
            rcnt[idx] = hi - lo
            rmean[idx] = np.where(hi - lo > 0, (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1), np.nan)
    return cnt, mean, rcnt, rmean


def add_history(query, hist, lead_hours=LEAD_HOURS):
    q = query.copy()
    c, m, rc, rm = _hist_stats(q, hist, ["movie_id"], "movie", lead_hours)
    q["movie_prior_shows"], q["movie_hist_mean_occ"] = c, m
    q["movie_recent7_count"], q["movie_recent7_mean_occ"] = rc, rm
    q["genre_hist_mean_occ"] = _hist_stats(
        q, hist, ["genre"], "genre", lead_hours, window_days=30)[3]  # trailing-30d genre mean
    h2, q2 = hist.assign(weekday=hist.show_datetime.dt.weekday, hour=hist.show_datetime.dt.hour), \
        q.assign(weekday=q.show_datetime.dt.weekday, hour=q.show_datetime.dt.hour)
    q["slot_hist_mean_occ"] = _hist_stats(q2, h2, ["weekday", "hour"], "slot", lead_hours)[1]
    return q


def build_features(df, history):
    """df: rows to featurise; history: cleaned table of already-observed shows."""
    return add_history(add_calendar(df), history)
