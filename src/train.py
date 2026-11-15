"""Repeatable training pipeline: clean -> features -> chronological split -> baselines ->
candidate models (tuned on VALIDATION only) -> one-shot TEST evaluation -> artifact.

Run:  python -m src.train            (add --fast for a quick smoke run)
"""
import argparse, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .features import CAT, NUM, FEATURES, TARGET, LEAD_HOURS, clean, build_features
from .generate_data import generate

SEED = 42
TRAIN_END, VAL_END = "2023-12-31", "2024-03-31"   # train <= 2023, val = Q1 2024, test = Q2 2024
BANDS = [0.0, 0.40, 0.75, 1.01]                    # staffing bands: low / medium / high occupancy


# ------------------------------------------------------------------ metrics
def metrics(y_true_occ, y_pred_occ, cap):
    yt, yp = y_true_occ * cap, np.clip(y_pred_occ, 0, 1) * cap   # back to ticket counts
    err = yp - yt
    band_t, band_p = np.digitize(y_true_occ, BANDS[1:-1]), np.digitize(np.clip(y_pred_occ, 0, 1), BANDS[1:-1])
    sold_out = y_true_occ >= 0.95
    return {
        "n": int(len(yt)),
        "MAE_tickets": float(np.mean(np.abs(err))),
        "RMSE_tickets": float(np.sqrt(np.mean(err ** 2))),
        "WAPE": float(np.abs(err).sum() / max(yt.sum(), 1e-9)),
        "MAE_occupancy": float(np.mean(np.abs(y_pred_occ.clip(0, 1) - y_true_occ))),
        "staffing_band_accuracy": float(np.mean(band_t == band_p)),
        "sellout_recall(pred>=85%)": float(np.mean(np.clip(y_pred_occ, 0, 1)[sold_out] >= 0.85)) if sold_out.any() else None,
        "underforecast_tickets_share": float(np.clip(-err, 0, None).sum() / max(yt.sum(), 1e-9)),
    }


def group_masks(X, train_movies):
    c = X["movie_prior_shows"]
    return {"all": np.ones(len(X), bool), "cold_start(0 prior shows)": (c == 0).values,
            "low_data(1-9 prior shows)": ((c >= 1) & (c < 10)).values,
            "established(>=10 prior shows)": (c >= 10).values,
            "movie_unseen_in_train": (~X["movie_id"].isin(train_movies)).values}


# ------------------------------------------------------------------ models
def baseline_movie(X, g):
    """Movie's own historical mean occupancy x capacity (fallback: genre, slot, global)."""
    p = X["movie_hist_mean_occ"].copy()
    for col in ["genre_hist_mean_occ", "slot_hist_mean_occ"]:
        p = p.fillna(X[col])
    return p.fillna(g).values


def baseline_slot(X, g):
    """Seasonal capacity-adjusted baseline: historical mean occupancy of the same weekday x
    show-hour slot (known at T-48h) x this screen's capacity."""
    return X["slot_hist_mean_occ"].fillna(g).values


BASELINES = {"baseline_movie_history": baseline_movie, "baseline_weekday_hour_slot": baseline_slot}


def make_pipe(kind, **p):
    if kind == "ridge":
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT),
                                 ("n", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                                                 ("s", StandardScaler())]), NUM)])
        return Pipeline([("pre", pre), ("m", Ridge(alpha=p["alpha"]))])
    if kind == "rf":
        pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), CAT),
                                 ("n", SimpleImputer(strategy="median", add_indicator=True), NUM)])
        return Pipeline([("pre", pre), ("m", RandomForestRegressor(
            n_estimators=p["n"], min_samples_leaf=p["leaf"], n_jobs=-1, random_state=SEED))])
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                             ("n", "passthrough", NUM)])   # HGB handles NaN (cold start) natively
    return Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(
        learning_rate=p["lr"], max_depth=p["depth"], max_iter=p["iter"],
        min_samples_leaf=p["leaf"], random_state=SEED))])


def grids(fast):
    if fast:
        return {"ridge": [dict(alpha=a) for a in (1, 10)], "rf": [dict(n=60, leaf=5)],
                "hgb": [dict(lr=0.1, depth=4, iter=100, leaf=20)]}
    return {"ridge": [dict(alpha=a) for a in (0.1, 1, 10, 100)],
            "rf": [dict(n=200, leaf=l) for l in (3, 10, 30)],
            "hgb": [dict(lr=lr, depth=d, iter=300, leaf=l)
                    for lr, d, l in itertools.product((0.03, 0.08), (3, 5), (20, 60))]}


def wape(model, X, y, cap):
    return metrics(y.values, model.predict(X[FEATURES]), cap.values)["WAPE"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shows.csv")
    ap.add_argument("--model-out", default="models/model.joblib")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--train-end", default=TRAIN_END); ap.add_argument("--val-end", default=VAL_END)
    a = ap.parse_args(argv)
    np.random.seed(SEED)
    rep = Path(a.reports); (rep / "figures").mkdir(parents=True, exist_ok=True)
    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)

    if not Path(a.data).exists():
        kw = {k: v for k, v in dict(start=a.start, end=a.end).items() if v}
        Path(a.data).parent.mkdir(parents=True, exist_ok=True)
        generate(seed=SEED, **kw).to_csv(a.data, index=False)
    df, dq = clean(pd.read_csv(a.data))
    X = build_features(df, df)                      # history features are strictly point-in-time
    dt = X["show_datetime"]
    t_end, v_end = pd.Timestamp(a.train_end) + pd.Timedelta(days=1), pd.Timestamp(a.val_end) + pd.Timedelta(days=1)
    tr, va, te = dt < t_end, (dt >= t_end) & (dt < v_end), dt >= v_end
    split = {"train": [str(dt[tr].min()), str(dt[tr].max()), int(tr.sum())],
             "validation": [str(dt[va].min()), str(dt[va].max()), int(va.sum())],
             "test": [str(dt[te].min()), str(dt[te].max()), int(te.sum())]}
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    train_movies = set(Xtr.movie_id)
    gmean = float(Xtr[TARGET].mean())

    results = {"val": {}, "test": {}}
    # --- baselines (no fitting, so nothing to tune) ---
    base = {"naive_global_mean": lambda Z: np.full(len(Z), gmean)}
    base.update({k: (lambda Z, f=f: f(Z, gmean)) for k, f in BASELINES.items()})
    for nm, f in base.items():
        for s, Z in (("val", Xva), ("test", Xte)):
            results[s][nm] = metrics(Z[TARGET].values, f(Z), Z.screen_capacity.values)

    # --- candidate models: choose hyper-parameters on VALIDATION WAPE only ---
    fitted, chosen = {}, {}
    for kind, plist in grids(a.fast).items():
        best = (9e9, None, None)
        for p in plist:
            m = make_pipe(kind, **p).fit(Xtr[FEATURES], Xtr[TARGET])
            w = wape(m, Xva, Xva[TARGET], Xva.screen_capacity)
            if w < best[0]:
                best = (w, p, m)
        chosen[kind], fitted[kind] = best[1], best[2]
        for s, Z in (("val", Xva), ("test", Xte)):
            results[s][kind] = metrics(Z[TARGET].values, best[2].predict(Z[FEATURES]), Z.screen_capacity.values)
    best_base = min(BASELINES, key=lambda k: results["val"][k]["WAPE"])   # strongest baseline (validation)
    cand = [k for k in fitted]
    best_name = min(cand, key=lambda k: results["val"][k]["WAPE"])   # selection by validation only
    model = fitted[best_name]

    # --- error analysis on TEST by data-availability group ---
    by_group = {}
    masks = group_masks(Xte, train_movies)
    for g, mk in masks.items():
        if mk.sum() == 0:
            continue
        Z = Xte[mk]
        by_group[g] = {
            "baseline": metrics(Z[TARGET].values, BASELINES[best_base](Z, gmean), Z.screen_capacity.values),
            best_name: metrics(Z[TARGET].values, model.predict(Z[FEATURES]), Z.screen_capacity.values)}

    # --- uncertainty: empirical validation-residual quantiles (occupancy) per data group ---
    res = Xva[TARGET].values - np.clip(model.predict(Xva[FEATURES]), 0, 1)
    vm = group_masks(Xva, train_movies)
    q = {}
    for g in ["cold_start(0 prior shows)", "low_data(1-9 prior shows)", "established(>=10 prior shows)", "all"]:
        r = res[vm[g]] if vm[g].sum() >= 30 else res
        q[g] = [float(np.quantile(r, 0.10)), float(np.quantile(r, 0.90))]
    cover = {}
    pt = np.clip(model.predict(Xte[FEATURES]), 0, 1)
    for g, key in [("cold", "cold_start(0 prior shows)"), ("low", "low_data(1-9 prior shows)"),
                   ("est", "established(>=10 prior shows)")]:
        mk = masks[key]
        if mk.sum():
            lo, hi = pt[mk] + q[key][0], pt[mk] + q[key][1]
            cover[g] = float(np.mean((Xte[TARGET].values[mk] >= lo) & (Xte[TARGET].values[mk] <= hi)))

    # --- permutation importance on TEST for chosen model ---
    pi = permutation_importance(model, Xte[FEATURES], Xte[TARGET], n_repeats=5,
                                random_state=SEED, scoring="neg_mean_absolute_error")
    imp = pd.DataFrame({"feature": FEATURES, "mae_increase": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("mae_increase", ascending=False)
    imp.to_csv(rep / "feature_importance.csv", index=False)

    # --- reference values for local explanations ---
    ref = {c: (Xtr[c].mode().iloc[0] if c in CAT else float(Xtr[c].median())) for c in FEATURES}
    ref["movie_hist_mean_occ"] = ref["movie_recent7_mean_occ"] = np.nan  # "no history" reference

    out = {"seed": SEED, "lead_hours": LEAD_HOURS, "split": split, "data_quality": dq,
           "chosen_hyperparameters": chosen, "best_model": best_name, "strongest_baseline": best_base, "global_train_mean_occ": gmean,
           "metrics": results, "test_by_group": by_group, "interval_coverage_test_80pct": cover}
    (rep / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    joblib.dump({"pipeline": model, "best_name": best_name, "features": FEATURES, "ref": ref,
                 "residual_q": q, "lead_hours": LEAD_HOURS, "global_mean": gmean,
                 "train_movies": sorted(train_movies), "chosen": chosen,
                 "importance": imp.head(10).to_dict("records")}, a.model_out)

    # --- figures ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    X.groupby("weekday")[TARGET].mean().plot.bar(ax=ax[0], title="Mean occupancy by weekday (0=Mon)")
    X.groupby("hour")[TARGET].mean().plot.bar(ax=ax[1], title="Mean occupancy by show hour")
    X.groupby(X.days_since_release.clip(upper=40) // 5 * 5)[TARGET].mean().plot(ax=ax[2], marker="o",
        title="Occupancy vs days since release"); plt.tight_layout()
    plt.savefig(rep / "figures/eda.png", dpi=110); plt.close()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Xte[TARGET], pt, s=4, alpha=.3); ax.plot([0, 1], [0, 1], "r--")
    ax.set(xlabel="actual occupancy", ylabel="predicted occupancy", title=f"Test: {best_name}")
    plt.savefig(rep / "figures/pred_vs_actual.png", dpi=110, bbox_inches="tight"); plt.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    gk = [g for g in by_group if g != "all"]
    ax.bar(np.arange(len(gk)) - .2, [by_group[g]["baseline"]["WAPE"] for g in gk], .4, label="baseline")
    ax.bar(np.arange(len(gk)) + .2, [by_group[g][best_name]["WAPE"] for g in gk], .4, label=best_name)
    ax.set_xticks(range(len(gk))); ax.set_xticklabels([g.split("(")[0] for g in gk], rotation=20)
    ax.set_ylabel("WAPE (test)"); ax.legend(); plt.tight_layout()
    plt.savefig(rep / "figures/wape_by_group.png", dpi=110); plt.close()

    print("Split:", json.dumps(split)); print("Data quality:", dq); print("Chosen:", chosen)
    print(pd.DataFrame(results["test"]).T[["MAE_tickets", "RMSE_tickets", "WAPE", "staffing_band_accuracy"]].round(3))
    print("Best by validation:", best_name); return out


if __name__ == "__main__":
    main()
