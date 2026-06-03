"""Vectors and dot product -- length, angle, and projection."""

import streamlit as st

from lib import components as C
from lib.linear_algebra import angle_between, dot, project_onto, vector_from_polar, vector_norm
from lib.plotting import vector_dot_figure

C.intro(
    "Vectors and dot product",
    "See how length and angle control whether two vectors point together or against each other.",
)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    col_a, col_b = st.columns(2)
    with col_a:
        length_a = st.slider("Length of vector a", 0.5, 5.0, 3.0, 0.1)
        angle_a = st.slider("Angle of vector a", 0, 360, 30, 5)
    with col_b:
        length_b = st.slider("Length of vector b", 0.5, 5.0, 3.0, 0.1)
        angle_b = st.slider("Angle of vector b", 0, 360, 70, 5)

a = vector_from_polar(length_a, angle_a)
b = vector_from_polar(length_b, angle_b)
projection = project_onto(a, b)
theta = angle_between(a, b)
dot_value = dot(a, b)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(vector_dot_figure(a, b, projection), width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Dot product", f"{dot_value:.2f}"),
    ("Angle", f"{theta:.0f} degrees"),
    ("Projection length", f"{vector_norm(projection):.2f}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("The dot product combines length with alignment:")
    st.latex(r"a \cdot b = \lVert a \rVert \lVert b \rVert \cos(\theta)")
    st.markdown("A positive dot product means the vectors point mostly together. A negative one means they point mostly apart.")
    st.markdown("Projection drops a perpendicular shadow from one vector onto the other.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Make the vectors point in the same direction.",
     "The dot product becomes large and positive."),
    ("Set the vectors about 90 degrees apart.",
     "The dot product gets close to zero because neither vector points along the other."),
    ("Point them in opposite directions.",
     "The dot product becomes negative."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "A dot product can be large because vectors are long, not because they are similar. "
    "Use cosine similarity when direction matters more than magnitude."
)
