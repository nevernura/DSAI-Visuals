"""Perceptron -- one threshold neuron."""

import numpy as np
import streamlit as st

from lib import components as C
from lib.data_options import CLASSIFICATION_OPTIONS, classification_data
from lib.models import fit_perceptron, perceptron_predict
from lib.plotting import perceptron_figure

C.intro(
    "Perceptron",
    "Train a single neuron that behaves like logistic regression with a hard cutoff.",
)

st.session_state.setdefault("perc_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", CLASSIFICATION_OPTIONS)
    learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
    n_steps = st.slider("Training passes", 5, 120, 60, 5)
    if st.button("New data"):
        st.session_state.perc_seed += 1

points, labels, data_note = classification_data(data_kind, seed=st.session_state.perc_seed, n=140)
weights, bias, mistakes = fit_perceptron(points, labels, learning_rate, n_steps)
predictions = perceptron_predict(points, weights, bias)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(perceptron_figure(points, labels, weights, bias), width="stretch")
st.caption(data_note)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Training accuracy", f"{np.mean(predictions == labels) * 100:.0f}%"),
    ("Final mistakes/pass", f"{mistakes[-1] if mistakes else 0}"),
    ("Passes run", f"{len(mistakes)}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("A perceptron computes a score and applies a hard threshold:")
    st.latex(r"\hat{y} = \mathbb{1}[w^	op x + b \ge 0]")
    st.markdown("When it gets a point wrong, it nudges the weights toward the correct side:")
    st.latex(r"w \leftarrow w + \eta y x")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Use **Linearly separable** data and increase **Training passes**.", "Mistakes often fall to zero."),
    ("Switch to **Non-linear rings**.", "One hard line cannot satisfy every point."),
    ("Change the **Learning rate**.", "Bigger nudges move faster but can bounce around."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "The perceptron only converges when the data is linearly separable. Overlap "
    "or curved classes can keep it updating forever."
)
