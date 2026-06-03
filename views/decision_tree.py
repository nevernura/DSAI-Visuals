"""Decision tree -- readable axis-aligned splits."""

import streamlit as st

from lib import components as C, datasets
from lib.models import accuracy, fit_decision_tree
from lib.plotting import classifier_figure

C.intro(
    "Decision tree",
    "Split the space with yes/no questions a beginner can read from the screen.",
)

st.session_state.setdefault("tree_seed", 0)

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    max_depth = st.slider("Max depth", 1, 8, 3)
    noisy = st.checkbox("Make the data noisy", value=False)
    if st.button("New data"):
        st.session_state.tree_seed += 1

# --- Data -----------------------------------------------------------------
spread = 1.9 if noisy else 1.25
points, labels = datasets.blobs(n=180, centers=2, spread=spread,
                                seed=st.session_state.tree_seed)
model = fit_decision_tree(points, labels, max_depth=max_depth)

# --- Visualisation --------------------------------------------------------
st.plotly_chart(classifier_figure(model, points, labels), width="stretch")

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
    st.markdown("The fitted rectangles are easy to inspect because every split is horizontal or vertical.")
    st.markdown("Depth controls how many questions the tree can ask before making a prediction.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Start with depth 1.",
     "The tree has only one split, so it draws one broad rule."),
    ("Increase depth slowly.",
     "More rectangles appear and the training accuracy rises."),
    ("Turn on **Make the data noisy** and push depth high.",
     "The tree starts carving tiny regions around individual points: visible overfitting."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "A deep tree can memorize noisy training data with many tiny rectangles. "
    "That can look impressive on the training set while generalizing poorly."
)
