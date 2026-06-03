"""K-means -- assign points, recenter clusters, repeat."""

import streamlit as st

from lib import components as C
from lib.clustering import inertia, run_kmeans
from lib.data_options import CLUSTERING_OPTIONS, clustering_data
from lib.plotting import kmeans_figure

C.intro(
    "K-means",
    "Watch clusters form by alternating between nearest-centroid assignment and recentering.",
)

st.session_state.setdefault("km_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", CLUSTERING_OPTIONS)
    bad_init = st.checkbox("Use bad initialization", value=False)
    if st.button("New data"):
        st.session_state.km_seed += 1
    points, data_note = clustering_data(data_kind, seed=st.session_state.km_seed)
    history = run_kmeans(points, k=3, bad_init=bad_init, max_steps=8)
    step = st.slider("Scrub assign/recenter steps", 0, len(history) - 1, 0)

phase, centroids, labels = history[step]
trail = [state[1] for state in history[:step + 1]]

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(kmeans_figure(points, centroids, labels, phase, trail=trail), width="stretch")
st.caption(data_note)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Step", f"{step} / {len(history) - 1}"),
    ("Phase", phase),
    ("Inertia", f"{inertia(points, labels, centroids):.0f}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("K-means repeats two simple operations:")
    st.latex(r"c_i = rg\min_k \lVert x_i - \mu_k Vert^2")
    st.latex(r"\mu_k = rac{1}{|C_k|}\sum_{x_i \in C_k} x_i")
    st.markdown("First assign each point, then move each centroid to the mean of its assigned points.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Drag the scrubber from left to right.", "Centroid trails show how cluster centers move each iteration."),
    ("Try **Rings**.", "Centroid-shaped clusters are a poor fit for ring-shaped data."),
    ("Turn on **Use bad initialization**.", "Bad starting centroids can converge to a worse local answer."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "K-means only improves from its starting centroids. A bad initialization can "
    "converge to a worse local answer, which is why real workflows try many starts."
)
