"""Eigenvectors and eigenvalues -- directions a matrix leaves aligned."""

import numpy as np
import streamlit as st

from lib import components as C
from lib.eigen import eigensystem, symmetric_transform
from lib.plotting import eigen_figure

C.intro(
    "Eigenvectors and eigenvalues",
    "Find the special directions that a matrix stretches without turning.",
)

st.session_state.setdefault("eig_a", 2.4)
st.session_state.setdefault("eig_b", 0.7)
st.session_state.setdefault("eig_angle", 35)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    show_circle = st.checkbox("Show transformed unit circle", value=True)
    if st.button("Make every direction special"):
        st.session_state.eig_a = 1.5
        st.session_state.eig_b = 1.5
    stretch_1 = st.slider("Stretch along direction 1", -3.0, 3.0, step=0.1, key="eig_a")
    stretch_2 = st.slider("Stretch along direction 2", -3.0, 3.0, step=0.1, key="eig_b")
    angle = st.slider("Rotate the special directions", 0, 180, step=5, key="eig_angle")

matrix = symmetric_transform(stretch_1, stretch_2, angle)
values, vectors = eigensystem(matrix)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(eigen_figure(matrix, values, vectors, show_circle=show_circle), width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Eigenvalue 1", f"{values[0]:.2f}"),
    ("Eigenvalue 2", f"{values[1]:.2f}"),
    ("Determinant", f"{np.linalg.det(matrix):.2f}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("An eigenvector keeps pointing along the same line after a matrix acts:")
    st.latex(r"A v = \lambda v")
    st.markdown("The vector is the direction. The eigenvalue is how much that direction stretches, flips, or collapses.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Drag **Rotate the special directions**.", "The dashed eigenvector lines rotate with the transform."),
    ("Make one stretch negative.", "That eigen-direction flips through the origin."),
    ("Press **Make every direction special**.", "A uniform scale leaves every direction special."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "When the eigenvalues are equal, the visualization stops having two clear "
    "special directions because every direction is preserved equally."
)
