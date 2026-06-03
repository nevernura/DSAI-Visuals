"""Decision tree -- readable axis-aligned splits."""

import streamlit as st

from lib import components as C
from lib.data_options import CLASSIFICATION_OPTIONS, classification_data
from lib.models import accuracy, fit_decision_tree
from lib.plotting import classifier_figure

C.intro(
    "Decision tree",
    "Split the space with yes/no questions a beginner can read from the screen.",
)

st.session_state.setdefault("tree_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", CLASSIFICATION_OPTIONS)
    max_depth = st.slider("Max depth", 1, 8, 3)
    if st.button("New data"):
        st.session_state.tree_seed += 1

points, labels, data_note = classification_data(data_kind, seed=st.session_state.tree_seed)
model = fit_decision_tree(points, labels, max_depth=max_depth)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(classifier_figure(model, points, labels), width="stretch")
st.caption(data_note)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Training accuracy", f"{accuracy(model, points, labels) * 100:.0f}%"),
    ("Leaves", f"{model.get_n_leaves()}"),
    ("Depth", f"{model.get_depth()}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("A tree asks one feature-threshold question at each split:")
    st.latex(r"x_j \le t")
    st.markdown("Every split is horizontal or vertical, so the rectangles are easy to inspect.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Start with depth 1.", "The tree has only one broad split."),
    ("Increase depth slowly.", "More rectangles appear and training accuracy rises."),
    ("Try **Overlapping** data and push depth high.", "The tree carves tiny regions around individual points."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "A deep tree can memorize noisy training data with many tiny rectangles, "
    "which may generalize poorly."
)
