"""Logistic regression -- classification with a fitted boundary.

Same recipe as linear_regression.py: intro -> controls -> visualisation ->
metrics -> math -> tasks -> break-it. The page delegates data generation,
plotting, and numeric work to shared lib modules.
"""

import numpy as np
import streamlit as st

from lib import components as C, datasets
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

# --- Controls (part 1): options that change the data -----------------------
with st.sidebar:
    st.header("Controls")
    show_probability = st.checkbox("Show probability surface", value=True)
    overlap = st.checkbox("Make classes overlap", value=False)
    if st.button("New data"):
        st.session_state.logr_seed += 1

# --- Data ------------------------------------------------------------------
spread = 2.2 if overlap else 1.2
points, labels = datasets.blobs(n=160, centers=2, spread=spread,
                                seed=st.session_state.logr_seed)
points = np.asarray(points, dtype=float)
labels = np.asarray(labels, dtype=int)

best_weights, best_bias = fit(points, labels)

# --- Controls (part 2): the classifier itself ------------------------------
with st.sidebar:
    if st.button("Fit best boundary"):
        st.session_state.logr_w1 = float(np.clip(round(best_weights[0], 2), -2.0, 2.0))
        st.session_state.logr_w2 = float(np.clip(round(best_weights[1], 2), -2.0, 2.0))
        st.session_state.logr_b = float(np.clip(round(best_bias, 2), -10.0, 10.0))
    w1 = st.slider("Weight for feature 1", -2.0, 2.0, step=0.05, key="logr_w1")
    w2 = st.slider("Weight for feature 2", -2.0, 2.0, step=0.05, key="logr_w2")
    bias = st.slider("Bias", -10.0, 10.0, step=0.1, key="logr_b")

weights = np.array([w1, w2], dtype=float)

# --- Visualisation ---------------------------------------------------------
st.plotly_chart(
    logistic_figure(points, labels, weights, bias,
                    show_probability=show_probability),
    width="stretch",
)

# --- Metrics ---------------------------------------------------------------
C.metric_row([
    ("Your accuracy", f"{accuracy(points, labels, weights, bias) * 100:.0f}%"),
    ("Your log loss", f"{log_loss(points, labels, weights, bias):.2f}"),
    ("Fitted log loss", f"{log_loss(points, labels, best_weights, best_bias):.2f}"),
])

# --- Maths -----------------------------------------------------------------
def _math():
    st.markdown("Logistic regression starts with a straight-line score:")
    st.latex(r"z = w_1x_1 + w_2x_2 + b")
    st.markdown("The sigmoid turns that score into a probability between 0 and 1:")
    st.latex(r"p(y=1\mid x) = \frac{1}{1 + e^{-z}}")
    st.markdown("The decision boundary is where the model is exactly unsure:")
    st.latex(r"p = 0.5 \quad\Longleftrightarrow\quad w_1x_1 + w_2x_2 + b = 0")
    st.markdown("Training chooses weights that reduce binary cross-entropy:")
    st.latex(r"-\frac{1}{n}\sum_i y_i\log(p_i) + (1-y_i)\log(1-p_i)")


C.show_math(_math)

# --- Guided tasks ----------------------------------------------------------
C.try_this([
    ("Press **Fit best boundary**.",
     "The orange line moves between the two clouds. Points on one side have "
     "high class-1 probability; points on the other side have low probability."),
    ("Drag the **Bias** slider.",
     "The boundary shifts without rotating. Bias controls where the cutoff sits "
     "after the feature weights choose a direction."),
    ("Tick **Make classes overlap**, then fit again.",
     "Some points are now mixed together. Even the fitted boundary has to accept "
     "mistakes because a single straight cut cannot separate every point."),
])

# --- The break-it moment ---------------------------------------------------
C.break_it(
    "When the classes overlap, logistic regression still draws one straight "
    "boundary. That makes it easy to interpret, but it cannot perfectly solve "
    "non-separable data without adding better features or a more flexible model."
)
