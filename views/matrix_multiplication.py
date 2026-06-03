"""Matrix multiplication -- composing transformations."""

import numpy as np
import streamlit as st

from lib import components as C
from lib.linear_algebra import named_matrix
from lib.plotting import matrix_multiplication_figure

C.intro(
    "Matrix multiplication",
    "See multiplication as doing one transformation, then another.",
)

PRESETS = ["Rotate 45 degrees", "Scale x", "Shear", "Reflect x"]

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    first_name = st.selectbox("First matrix", PRESETS, index=0)
    second_name = st.selectbox("Second matrix", PRESETS, index=1)

first = named_matrix(first_name)
second = named_matrix(second_name)
combined = second @ first

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(matrix_multiplication_figure(first, second), width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("det(first)", f"{np.linalg.det(first):.2f}"),
    ("det(second)", f"{np.linalg.det(second):.2f}"),
    ("det(product)", f"{np.linalg.det(combined):.2f}"),
])

st.caption("The right panel shows the product as: second matrix x first matrix.")

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("Matrix multiplication means composition:")
    st.latex(r"(B A)x = B(Ax)")
    st.markdown("The matrix closest to the vector acts first. That is why order matters.")
    st.latex(r"A B \ne B A \quad \text{in general}")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Choose rotate first, then scale.",
     "The grid rotates before it stretches."),
    ("Swap the two choices.",
     "The final grid usually changes. Matrix multiplication is order-sensitive."),
    ("Use a reflection.",
     "The determinant becomes negative, showing orientation flipped."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "You cannot usually swap two matrices and expect the same result. "
    "Changing order changes the story of what happened to the space."
)
