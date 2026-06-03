"""K-means -- assign points, recenter clusters, repeat."""

import streamlit as st

from lib import components as C, datasets
from lib.clustering import inertia, run_kmeans
from lib.plotting import kmeans_figure

C.intro(
    "K-means",
    "Watch clusters form by alternating between nearest-centroid assignment and recentering.",
)

st.session_state.setdefault("km_seed", 0)

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    bad_init = st.checkbox("Use bad initialization", value=False)
    if st.button("New data"):
        st.session_state.km_seed += 1

# --- Data -----------------------------------------------------------------
points, _ = datasets.blobs(n=180, centers=3, spread=1.05, seed=st.session_state.km_seed)
history = run_kmeans(points, k=3, bad_init=bad_init, max_steps=8)

with st.sidebar:
    step = st.slider("Scrub assign/recenter steps", 0, len(history) - 1, 0)

phase, centroids, labels = history[step]

# --- Visualisation --------------------------------------------------------
st.plotly_chart(kmeans_figure(points, centroids, labels, phase), width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Step", f"{step} / {len(history) - 1}"),
    ("Phase", phase),
    ("Inertia", f"{inertia(points, labels, centroids):.0f}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("K-means repeats two simple operations:")
    st.latex(r"c_i = \arg\min_k \lVert x_i - \mu_k \rVert^2")
    st.latex(r"\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i")
    st.markdown("First assign each point to its nearest centroid, then move each centroid to the mean of its assigned points.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Drag the scrubber from left to right.",
     "The plot alternates between assigning points and recentering the cluster markers."),
    ("Press **New data** a few times.",
     "Different clouds can converge in different numbers of steps."),
    ("Turn on **Use bad initialization**.",
     "Starting centroids in one area can lead to a worse final split even though the algorithm followed its rules."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "K-means only improves from its starting centroids. A bad initialization can "
    "converge to a worse local answer, which is why real workflows often try many starts."
)
