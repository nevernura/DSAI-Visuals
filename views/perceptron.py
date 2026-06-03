"""Perceptron -- one threshold neuron."""

import numpy as np
import streamlit as st

from lib import components as C, datasets
from lib.models import fit_perceptron, perceptron_predict
from lib.plotting import perceptron_figure

C.intro(
    "Perceptron",
    "Train a single neuron that behaves like logistic regression with a hard cutoff.",
)

st.session_state.setdefault("perc_seed", 0)

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
    n_steps = st.slider("Training passes", 5, 120, 60, 5)
    overlap = st.checkbox("Make data overlap", value=False)
    if st.button("New data"):
        st.session_state.perc_seed += 1

# --- Data -----------------------------------------------------------------
spread = 1.8 if overlap else 1.0
points, labels = datasets.blobs(n=140, centers=2, spread=spread, seed=st.session_state.perc_seed)
weights, bias, mistakes = fit_perceptron(points, labels, learning_rate, n_steps)
predictions = perceptron_predict(points, weights, bias)

# --- Visualisation --------------------------------------------------------
st.plotly_chart(perceptron_figure(points, labels, weights, bias), width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Training accuracy", f"{np.mean(predictions == labels) * 100:.0f}%"),
    ("Final mistakes/pass", f"{mistakes[-1] if mistakes else 0}"),
    ("Passes run", f"{len(mistakes)}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("A perceptron computes a score and applies a hard threshold:")
    st.latex(r"\hat{y} = \mathbb{1}[w^\top x + b \ge 0]")
    st.markdown("When it gets a point wrong, it nudges the weights toward the correct side:")
    st.latex(r"w \leftarrow w + \eta y x")
    st.markdown("This is the single-neuron ancestor of logistic regression and neural networks.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Leave the data separable and increase **Training passes**.",
     "The mistakes usually fall to zero once the boundary finds a separating line."),
    ("Turn on **Make data overlap**.",
     "The perceptron keeps making mistakes because no hard line can satisfy every point."),
    ("Change the **Learning rate**.",
     "Bigger nudges move the boundary faster but can bounce around on messy data."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "The perceptron only converges when the data is linearly separable. Overlap "
    "or curved classes can keep it updating forever."
)
