"""Synthetic cinema show-level data generator (fully reproducible, seeded).

Demand is simulated as a latent quantity that depends on movie quality/hype,
genre, weekday, show hour, holidays, days since release (decay) and price.
Observed tickets_sold = min(latent demand, screen capacity)  -> demand is
CENSORED at capacity, exactly like real box-office data.
Deliberate data-quality problems (missing prices, over-capacity counts,
duplicates, negative counts) are injected so that cleaning is meaningful.
No personal data is generated.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Family", "Horror"]
LANGS = ["Hindi", "English", "Punjabi", "Tamil"]
GENRE_EFF = dict(zip(GENRES, [0.15, 0.10, -0.10, 0.05, -0.05, 0.12, 0.0]))
LANG_EFF = dict(zip(LANGS, [0.10, 0.0, 0.05, -0.10]))
SCREENS = {1: 240, 2: 180, 3: 150, 4: 120, 5: 90}  # screen_id -> seats
SLOT_HOURS = [10, 13, 16, 19, 22]
WEEKDAY_EFF = [-0.25, -0.30, -0.20, -0.10, 0.15, 0.40, 0.35]  # Mon..Sun
HOUR_EFF = {10: -0.45, 13: -0.15, 16: 0.0, 19: 0.30, 22: 0.10}
HOLIDAYS = pd.to_datetime([
    "2023-01-26", "2023-03-08", "2023-04-14", "2023-08-15", "2023-10-02",
    "2023-10-24", "2023-11-12", "2023-12-25", "2024-01-26", "2024-03-25",
    "2024-04-14", "2024-05-01"])
START, END = "2023-01-01", "2024-06-30"


def make_movies(rng, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    fridays = pd.date_range(start - pd.Timedelta(days=28), end, freq="W-FRI")
    rows = []
    for f in fridays:
        for _ in range(rng.integers(1, 3)):
            buzz = float(np.clip(rng.beta(2.2, 2.2) * 100, 1, 99))
            quality = rng.normal(0, 0.5)               # hidden from the model
            hype = 0.012 * (buzz - 50) + quality       # buzz is informative, not perfect
            rows.append(dict(
                movie_id=f"M{len(rows):03d}", genre=rng.choice(GENRES),
                language=rng.choice(LANGS, p=[0.4, 0.3, 0.2, 0.1]),
                runtime_min=int(np.clip(rng.normal(135, 20), 85, 190)),
                is_franchise=int(rng.random() < 0.25), pre_release_buzz=round(buzz, 1),
                release_date=f.normalize(), hype=hype,
                run_days=int(np.clip(21 + 10 * quality + rng.normal(0, 4), 10, 49)),
                tau=float(np.clip(12 + 6 * quality + rng.normal(0, 2), 5, 25))))
    m = pd.DataFrame(rows)
    m.loc[m.is_franchise == 1, "hype"] += 0.3
    return m


def generate(start=START, end=END, seed=42):
    rng = np.random.default_rng(seed)
    movies = make_movies(rng, start, end)
    recs = []
    for day in pd.date_range(start, end):
        active = movies[(movies.release_date <= day) &
                        (day < movies.release_date + pd.to_timedelta(movies.run_days, "D"))]
        if active.empty:
            continue
        dse = (day - active.release_date).dt.days.values
        w = np.exp(active.hype.values - dse / (active.tau.values * 2.0))
        for sid, cap in SCREENS.items():
            p = w ** (cap / 150.0)
            p = p / p.sum()
            for hr in SLOT_HOURS:
                i = rng.choice(len(active), p=p)
                mv = active.iloc[i]
                days_since = (day - mv.release_date).days
                hol = int(day in HOLIDAYS)
                price = float(np.clip(140 + 25 * (cap > 200) + 20 * (hr >= 19)
                                      + 25 * (day.weekday() >= 4) + rng.normal(0, 12), 90, 400))
                log_mult = (mv.hype + GENRE_EFF[mv.genre] + LANG_EFF[mv.language]
                            + WEEKDAY_EFF[day.weekday()] + HOUR_EFF[hr] + 0.35 * hol
                            - days_since / mv.tau + (0.45 if days_since < 3 else 0.0)
                            - 0.004 * (price - 170))
                lam = cap * 0.55 * np.exp(log_mult)
                demand = rng.poisson(lam * rng.gamma(25, 1 / 25))
                recs.append(dict(
                    movie_id=mv.movie_id, genre=mv.genre, language=mv.language,
                    runtime_min=mv.runtime_min, is_franchise=mv.is_franchise,
                    pre_release_buzz=mv.pre_release_buzz, release_date=mv.release_date,
                    screen_id=sid, screen_capacity=cap,
                    show_datetime=day + pd.Timedelta(hours=hr),
                    ticket_price=round(price, 0), tickets_sold=int(min(demand, cap))))
    df = pd.DataFrame(recs)
    df.insert(0, "show_id", [f"S{i:06d}" for i in range(len(df))])
    # ---- inject data-quality problems (documented in the report) ----
    n = len(df)
    df.loc[rng.choice(n, int(0.015 * n), replace=False), "ticket_price"] = np.nan
    df.loc[rng.choice(n, int(0.003 * n), replace=False), "tickets_sold"] += 40   # over capacity
    df.loc[rng.choice(n, 8, replace=False), "tickets_sold"] = -1                  # invalid
    df = pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)     # duplicates
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shows.csv")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = generate(a.start, a.end, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
