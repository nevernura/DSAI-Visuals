"""SVM -- maximum-margin boundaries and the kernel trick."""

import streamlit as st

from lib import components as C, datasets
from lib.models import accuracy, fit_svm
from lib.plotting import classifier_figure

C.intro(
    "SVM",
    "Find a boundary with the widest safety margin, then use a kernel when straight lines fail.",
)

st.session_state.setdefault("svm_seed", 0)

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    dataset = st.radio("Dataset", ["separable blobs", "non-separable rings"])
    kernel = st.radio("Kernel", ["linear", "rbf"], horizontal=True)
    c_value = st.slider("Margin strictness (C)", 0.1, 5.0, 1.0, 0.1)
    gamma = st.slider("Kernel reach (gamma)", 0.1, 3.0, 1.0, 0.1)
    if st.button("New data"):
        st.session_state.svm_seed += 1

# --- Data -----------------------------------------------------------------
if dataset == "separable blobs":
    points, labels = datasets.blobs(n=160, centers=2, spread=1.15,
                                    seed=st.session_state.svm_seed)
else:
    points, labels = datasets.two_rings(n=180, noise=0.11,
                                        seed=st.session_state.svm_seed)

model = fit_svm(points, labels, kernel=kernel, c=c_value, gamma=gamma)

# --- Visualisation --------------------------------------------------------
st.plotly_chart(
    classifier_figure(model, points, labels, support_vectors=model.support_vectors_),
    width="stretch",
)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Training accuracy", f"{accuracy(model, points, labels) * 100:.0f}%"),
    ("Support vectors", f"{len(model.support_vectors_)}"),
    ("Kernel", kernel.upper()),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("A linear SVM chooses the separating line with the widest margin:")
    st.latex(r"y_i(w^\top x_i + b) \ge 1")
    st.markdown("The kernel trick lets the model act as if points were lifted into a richer space:")
    st.latex(r"K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)")
    st.markdown("The RBF kernel makes nearby points similar, which lets rings separate without drawing a straight line.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Use **non-separable rings** with the **linear** kernel.",
     "A straight boundary cannot cleanly separate the inner and outer rings."),
    ("Switch to the **rbf** kernel.",
     "The boundary bends around the inner ring. That is the kernel trick payoff."),
    ("Lower **Margin strictness (C)**.",
     "The SVM accepts more mistakes to keep a wider, smoother margin."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "A flexible kernel can fit shapes a line cannot, but high C and high gamma "
    "can make the boundary chase noise instead of the pattern."
)
