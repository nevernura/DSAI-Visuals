"""PCA -- principal directions and projection shadows."""

import streamlit as st

from lib import components as C
from lib.data_options import PCA_OPTIONS, pca_data
from lib.pca import fit_pca
from lib.plotting import pca_figure

C.intro(
    "PCA",
    "Spin a 3D cloud and watch its shadow fall onto the directions of most variation.",
)

st.session_state.setdefault("pca_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", PCA_OPTIONS)
    n_components = st.slider("Projection dimensions", 1, 2, value=2)
    show_shadow = st.checkbox("Show projection shadow", value=True)
    if st.button("New cloud"):
        st.session_state.pca_seed += 1

points, data_note = pca_data(data_kind, seed=st.session_state.pca_seed)
mean, components, variances, explained, scores = fit_pca(points)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(
    pca_figure(points, mean, components, scores, n_components=n_components, show_shadow=show_shadow),
    width="stretch",
)
st.caption(data_note)

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
    st.latex(r"\Sigma = rac{1}{n-1} X^	op X")
    st.latex(r"\Sigma v = \lambda v")
    st.markdown("The largest eigenvalue points to the direction with the most variance.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Rotate the 3D plot with your mouse.", "The long orange PC1 axis follows the longest direction."),
    ("Switch **Projection dimensions** from 2 to 1.", "The shadow collapses to a line."),
    ("Try a real dataset.", "PCA still finds directions of maximum spread."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "PCA keeps variance, not meaning. If the important signal lives in a small "
    "low-variance direction, PCA can hide it."
)
