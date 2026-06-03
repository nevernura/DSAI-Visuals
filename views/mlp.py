"""MLP -- stacking neurons to bend a boundary."""

import streamlit as st

from lib import components as C
from lib.data_options import CLASSIFICATION_OPTIONS, classification_data
from lib.models import accuracy, fit_mlp
from lib.plotting import classifier_figure

C.intro(
    "MLP",
    "Stack simple neurons into a hidden layer that can bend a classification boundary.",
)

st.session_state.setdefault("mlp_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", CLASSIFICATION_OPTIONS, index=2)
    hidden_units = st.slider("Hidden neurons", 2, 16, 8, 2)
    alpha = st.slider("Regularization", 0.0001, 0.02, 0.001, 0.0001, format="%.4f")
    if st.button("New data"):
        st.session_state.mlp_seed += 1

points, labels, data_note = classification_data(data_kind, seed=st.session_state.mlp_seed)
model = fit_mlp(points, labels, hidden_units=hidden_units, alpha=alpha)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(classifier_figure(model, points, labels), width="stretch")
st.caption(data_note)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Training accuracy", f"{accuracy(model, points, labels) * 100:.0f}%"),
    ("Hidden neurons", f"{hidden_units}"),
    ("Training iterations", f"{model.n_iter_}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("An MLP stacks neuron activations in layers:")
    st.latex(r"h = 	anh(W_1x + b_1)")
    st.latex(r"\hat{y} = \sigma(W_2h + b_2)")
    st.markdown("The hidden layer creates intermediate features, so the final boundary can curve.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Start with only two hidden neurons.", "The boundary may be too simple for rings."),
    ("Increase **Hidden neurons**.", "The model gains enough flexibility to bend around classes."),
    ("Raise **Regularization**.", "The model is discouraged from wiggly boundaries."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "More hidden units add flexibility, but they also make it easier to fit noise. "
    "Regularization keeps the boundary from becoming needlessly complex."
)
