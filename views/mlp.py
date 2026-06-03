"""MLP -- stacking neurons to bend a boundary."""

import streamlit as st

from lib import components as C, datasets
from lib.models import accuracy, fit_mlp
from lib.plotting import classifier_figure

C.intro(
    "MLP",
    "Stack simple neurons into a hidden layer that can bend a classification boundary.",
)

st.session_state.setdefault("mlp_seed", 0)

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    hidden_units = st.slider("Hidden neurons", 2, 16, 8, 2)
    alpha = st.slider("Regularization", 0.0001, 0.02, 0.001, 0.0001,
                      format="%.4f")
    if st.button("New data"):
        st.session_state.mlp_seed += 1

# --- Data -----------------------------------------------------------------
points, labels = datasets.two_rings(n=180, noise=0.12, seed=st.session_state.mlp_seed)
model = fit_mlp(points, labels, hidden_units=hidden_units, alpha=alpha)

# --- Visualisation --------------------------------------------------------
st.plotly_chart(classifier_figure(model, points, labels), width="stretch")

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Training accuracy", f"{accuracy(model, points, labels) * 100:.0f}%"),
    ("Hidden neurons", f"{hidden_units}"),
    ("Training iterations", f"{model.n_iter_}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("An MLP stacks neuron activations in layers:")
    st.latex(r"h = \tanh(W_1x + b_1)")
    st.latex(r"\hat{y} = \sigma(W_2h + b_2)")
    st.markdown("The hidden layer creates intermediate features, so the final boundary can curve around non-separable data.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Start with only two hidden neurons.",
     "The boundary may be too simple to wrap around the rings."),
    ("Increase **Hidden neurons**.",
     "The boundary gains enough flexibility to separate the inner and outer classes."),
    ("Raise **Regularization**.",
     "The model is discouraged from making an overly wiggly boundary."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "More hidden units add flexibility, but they also make it easier to fit noise. "
    "Regularization keeps the boundary from becoming needlessly complex."
)
