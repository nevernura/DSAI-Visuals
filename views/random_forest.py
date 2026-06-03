"""Random forest -- many small trees voting together."""

import streamlit as st

from lib import components as C
from lib.data_options import CLASSIFICATION_OPTIONS, classification_data
from lib.models import accuracy, fit_random_forest
from lib.plotting import classifier_figure

C.intro(
    "Random forest",
    "Let many small decision trees vote instead of trusting one brittle tree.",
)

st.session_state.setdefault("rf_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", CLASSIFICATION_OPTIONS, index=2)
    n_trees = st.slider("Trees", 3, 31, 15, 2)
    max_depth = st.slider("Max depth per tree", 1, 7, 4)
    if st.button("New data"):
        st.session_state.rf_seed += 1

points, labels, data_note = classification_data(data_kind, seed=st.session_state.rf_seed)
model = fit_random_forest(points, labels, n_estimators=n_trees, max_depth=max_depth)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(classifier_figure(model, points, labels), width="stretch")
st.caption(data_note)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Training accuracy", f"{accuracy(model, points, labels) * 100:.0f}%"),
    ("Trees voting", f"{n_trees}"),
    ("Average leaves", f"{sum(t.get_n_leaves() for t in model.estimators_) / n_trees:.1f}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("A random forest averages many tree predictions:")
    st.latex(r"\hat{p}(y=1|x) = rac{1}{T}\sum_{t=1}^{T} \hat{p}_t(y=1|x)")
    st.markdown("Each tree sees a bootstrap sample and random feature choices, so their mistakes are less synchronized.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Compare a few trees with many trees.", "More trees smooth the vote."),
    ("Raise **Max depth per tree**.", "The boundary becomes more flexible."),
    ("Try **Overlapping** data.", "Noise can still make the forest boundary jagged."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "Random forests are robust, not magic. Deep trees on noisy data can still "
    "make the forest overly jagged."
)
