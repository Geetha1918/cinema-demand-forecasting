"""Interactive app.  Run:  streamlit run app/streamlit_app.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from src.predict import Forecaster
from src.generate_data import GENRES, LANGS, SCREENS

st.set_page_config(page_title="Cinema Demand Forecast", layout="centered")
st.title("🎬 Cinema Ticket Demand Forecast")
st.caption("Decision support only - a human must review before allocating screens or staff.")


@st.cache_resource
def load():
    return Forecaster()


f = load()
movies = f.history.drop_duplicates("movie_id").set_index("movie_id")
mode = st.radio("Movie", ["Existing movie in history", "Brand-new movie (cold start)"], horizontal=True)
if mode.startswith("Existing"):
    mid = st.selectbox("Movie", list(movies.index)[::-1])
    m = movies.loc[mid]
    meta = dict(movie_id=mid, genre=m.genre, language=m.language, runtime_min=int(m.runtime_min),
                is_franchise=int(m.is_franchise), pre_release_buzz=float(m.pre_release_buzz),
                release_date=str(m.release_date.date()))
else:
    c1, c2 = st.columns(2)
    meta = dict(movie_id="NEW", genre=c1.selectbox("Genre", GENRES), language=c2.selectbox("Language", LANGS),
                runtime_min=c1.number_input("Runtime (min)", 60, 240, 130), is_franchise=int(c2.checkbox("Franchise / sequel")),
                pre_release_buzz=c1.slider("Pre-release buzz (0-100)", 0.0, 100.0, 50.0),
                release_date=str(c2.date_input("Release date", pd.Timestamp(f.last_history).date()))) 
d1, d2, d3, d4 = st.columns(4)
day = d1.date_input("Show date", pd.Timestamp(f.last_history).date())
hour = d2.selectbox("Start hour", [10, 13, 16, 19, 22], index=3)
cap = d3.selectbox("Screen capacity", sorted(set(SCREENS.values())), index=2)
price = d4.number_input("Ticket price", 50.0, 600.0, 180.0)
if st.button("Forecast", type="primary"):
    r = f.predict({**meta, "screen_capacity": int(cap), "ticket_price": float(price),
                   "show_datetime": f"{day}T{hour:02d}:00:00"})
    a, b = st.columns(2)
    a.metric("Forecast tickets", r["predicted_tickets"], f"{r['predicted_occupancy']:.0%} of {cap} seats")
    b.metric("80% interval", f"{r['interval_80pct_tickets'][0]:.0f} - {r['interval_80pct_tickets'][1]:.0f}")
    st.write(f"**Data availability:** {r['data_availability_group']} ({r['movie_prior_shows_known']} prior shows known) | model: `{r['model']}`")
    for w in r["warnings"]:
        st.warning(w)
    st.subheader("Why this forecast? (what-if effect of each factor, in tickets)")
    st.bar_chart(pd.DataFrame(r["top_factors"]).set_index("factor"))
    st.caption(r["note"])
