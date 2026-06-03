"""Orthogonality and projections -- shortest shadows onto a line."""

import streamlit as st

from lib import components as C
from lib.linear_algebra import dot, project_onto, vector_from_polar, vector_norm
from lib.plotting import projection_figure

C.intro(
    "Orthogonality and projections",
    "Drop a perpendicular shadow from a vector onto a line.",
)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    vector_length = st.slider("Vector length", 0.5, 5.0, 3.8, 0.1)
    vector_angle = st.slider("Vector angle", 0, 360, 50, 5)
    line_angle = st.slider("Target line angle", 0, 180, 15, 5)

vector = vector_from_polar(vector_length, vector_angle)
direction = vector_from_polar(1.0, line_angle)
projection = project_onto(vector, direction)
residual = vector - projection

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(projection_figure(vector, direction, projection), width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Projection length", f"{vector_norm(projection):.2f}"),
    ("Residual length", f"{vector_norm(residual):.2f}"),
    ("Residual dot line", f"{dot(residual, direction):.2f}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("Projection keeps the part of a vector that points along a direction:")
    st.latex(r"\operatorname{proj}_b(a) = \frac{a \cdot b}{b \cdot b}b")
    st.markdown("The leftover residual is orthogonal to the target line:")
    st.latex(r"(a - \operatorname{proj}_b(a)) \cdot b = 0")
    st.markdown("This is the geometry behind least squares and PCA shadows.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Align the vector with the target line.",
     "The residual shrinks toward zero because the whole vector is already on the line."),
    ("Make the vector perpendicular to the line.",
     "The projection shrinks toward zero."),
    ("Watch **Residual dot line**.",
     "It stays near zero because the residual is perpendicular to the line."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "Projection is the closest point on the line, but only under ordinary L2 distance. "
    "Change the distance rule and the best shadow can change too."
)
