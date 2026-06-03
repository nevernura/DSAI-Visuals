"""Matrix transformations -- rotate, scale, and shear a grid."""

import numpy as np
import streamlit as st

from lib import components as C
from lib.linear_algebra import transform_matrix
from lib.plotting import matrix_transform_figure

C.intro(
    "Matrix transformations",
    "Watch a matrix move the whole coordinate grid at once.",
)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    show_original = st.checkbox("Show original grid", value=True)
    rotation = st.slider("Rotation", -180, 180, 25, 5)
    scale_x = st.slider("Scale x", -3.0, 3.0, 1.4, 0.1)
    scale_y = st.slider("Scale y", -3.0, 3.0, 0.8, 0.1)
    shear_x = st.slider("Shear x", -2.0, 2.0, 0.3, 0.1)

matrix = transform_matrix(scale_x=scale_x, scale_y=scale_y,
                          shear_x=shear_x, angle_degrees=rotation)
determinant = float(np.linalg.det(matrix))

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(matrix_transform_figure(matrix, show_original=show_original),
                  width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Determinant", f"{determinant:.2f}"),
    ("Area scale", f"{abs(determinant):.2f}x"),
    ("Invertible", "yes" if abs(determinant) > 1e-9 else "no"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("A matrix transforms every point by multiplying it:")
    st.latex(r"x' = A x")
    st.markdown("The transformed basis vectors are the columns of the matrix.")
    st.markdown("The determinant tells how much area scales, and its sign tells whether orientation flips.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Set one scale close to zero.",
     "The grid collapses toward a line, and the matrix stops being invertible."),
    ("Make one scale negative.",
     "The grid flips across an axis, so the determinant changes sign."),
    ("Move the shear slider.",
     "Rectangles slant into parallelograms while area stays tied to the determinant."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "When the determinant is zero, the matrix squashes 2D space into a lower-dimensional shape. "
    "Information is lost, so there is no inverse that can undo it."
)
