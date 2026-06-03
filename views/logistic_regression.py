"""Logistic regression -- classification with a fitted boundary."""

import numpy as np
import streamlit as st

from lib import components as C
from lib.data_options import CLASSIFICATION_OPTIONS, classification_data
from lib.logistic import accuracy, fit, log_loss
from lib.plotting import logistic_figure

C.intro(
    "Logistic regression",
    "Draw a boundary that turns a point's position into a class probability.",
)

st.session_state.setdefault("logr_w1", -0.6)
st.session_state.setdefault("logr_w2", 0.0)
st.session_state.setdefault("logr_b", 0.0)
st.session_state.setdefault("logr_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", CLASSIFICATION_OPTIONS)
    show_probability = st.checkbox("Show probability surface", value=True)
    if st.button("New data"):
        st.session_state.logr_seed += 1

    points, labels, data_note = classification_data(data_kind, seed=st.session_state.logr_seed, n=160)
    st.caption(data_note)
    best_weights, best_bias = fit(points, labels)

    if st.button("Fit best boundary"):
        st.session_state.logr_w1 = float(np.clip(round(best_weights[0], 2), -3.0, 3.0))
        st.session_state.logr_w2 = float(np.clip(round(best_weights[1], 2), -3.0, 3.0))
        st.session_state.logr_b = float(np.clip(round(best_bias, 2), -10.0, 10.0))
    w1 = st.slider("Weight for feature 1", -3.0, 3.0, step=0.05, key="logr_w1")
    w2 = st.slider("Weight for feature 2", -3.0, 3.0, step=0.05, key="logr_w2")
    bias = st.slider("Bias", -10.0, 10.0, step=0.1, key="logr_b")

weights = np.array([w1, w2], dtype=float)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(
    logistic_figure(points, labels, weights, bias, show_probability=show_probability),
    width="stretch",
)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Your accuracy", f"{accuracy(points, labels, weights, bias) * 100:.0f}%"),
    ("Your log loss", f"{log_loss(points, labels, weights, bias):.2f}"),
    ("Fitted log loss", f"{log_loss(points, labels, best_weights, best_bias):.2f}"),
])

# --- Maths ----------------------------------------------------------------

def _math():
    st.markdown("Logistic regression starts with a straight-line score:")
    st.latex(r"z = w_1x_1 + w_2x_2 + b")
    st.markdown("The sigmoid turns that score into a probability between 0 and 1:")
    st.latex(r"p(y=1\mid x) = \frac{1}{1 + e^{-z}}")
    st.markdown("The decision boundary is where the model is exactly unsure:")
    st.latex(r"p = 0.5 \quad\Longleftrightarrow\quad w_1x_1 + w_2x_2 + b = 0")

C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Press **Fit best boundary**.",
     "The orange line moves between the two clouds."),
    ("Switch to **Non-linear rings**.",
     "A single straight boundary cannot wrap around the inner ring."),
    ("Drag the **Bias** slider.",
     "The boundary shifts without rotating."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "Logistic regression draws one straight boundary. It is easy to interpret, "
    "but it cannot perfectly solve non-separable data without better features."
)
