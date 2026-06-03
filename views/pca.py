"""PCA -- principal directions and projection shadows."""

import streamlit as st

from lib import components as C, datasets
from lib.pca import fit_pca
from lib.plotting import pca_figure

C.intro(
    "PCA",
    "Spin a 3D cloud and watch its shadow fall onto the directions of most variation.",
)

st.session_state.setdefault("pca_seed", 0)

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    n_components = st.slider("Projection dimensions", 1, 2, value=2)
    show_shadow = st.checkbox("Show projection shadow", value=True)
    if st.button("New cloud"):
        st.session_state.pca_seed += 1

# --- Data -----------------------------------------------------------------
points = datasets.correlated_cloud_3d(seed=st.session_state.pca_seed)
mean, components, variances, explained, scores = fit_pca(points)

# --- Visualisation --------------------------------------------------------
st.plotly_chart(
    pca_figure(points, mean, components, scores,
               n_components=n_components, show_shadow=show_shadow),
    width="stretch",
)

# --- Metrics --------------------------------------------------------------
shown = explained[:n_components].sum()
C.metric_row([
    ("PC1 variance", f"{explained[0] * 100:.0f}%"),
    ("PC2 variance", f"{explained[1] * 100:.0f}%"),
    ("Shown in shadow", f"{shown * 100:.0f}%"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("PCA centers the data, builds a covariance matrix, then finds its eigenvectors:")
    st.latex(r"\Sigma = \frac{1}{n-1} X^\top X")
    st.latex(r"\Sigma v = \lambda v")
    st.markdown(
        "The largest eigenvalue points to the direction with the most variance. "
        "Projecting onto the first few eigenvectors gives a lower-dimensional "
        "shadow that keeps as much spread as possible."
    )


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Rotate the 3D plot with your mouse.",
     "The long orange PC1 axis follows the longest direction through the cloud."),
    ("Switch **Projection dimensions** from 2 to 1.",
     "The shadow collapses to a line, keeping PC1 and discarding the next direction."),
    ("Press **New cloud** a few times.",
     "The exact points change, but PCA keeps finding the directions with the most spread."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "PCA keeps variance, not meaning. If the important signal lives in a small "
    "low-variance direction, PCA can hide it while preserving a very accurate-looking shadow."
)
