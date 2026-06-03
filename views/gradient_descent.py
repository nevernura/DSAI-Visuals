"""Gradient descent -- the reference module.

This page is the template made concrete. Read it top to bottom and you can see
the recipe: intro -> controls -> visualisation -> metrics -> math -> tasks ->
break-it. Copy this file to start any new module.
"""

import streamlit as st

from lib import components as C
from lib.descent import descend, loss, X_RANGE, Y_RANGE
from lib.plotting import descent_figure

C.intro(
    "Gradient descent",
    "Watch an optimizer roll downhill to the lowest loss -- and watch it fly "
    "off the rails when the steps get too big.",
)

# --- Controls -------------------------------------------------------------
if "gd_sx" not in st.session_state:
    st.session_state.gd_sx = -8.0
if "gd_sy" not in st.session_state:
    st.session_state.gd_sy = 6.0

with st.sidebar:
    st.header("Controls")
    lr = st.slider("Learning rate (\u03b7)", 0.01, 2.5, 0.30, 0.01)
    n_steps = st.slider("Max steps", 5, 80, 40, 5)
    st.caption("Starting point")
    preset = st.columns(3)
    if preset[0].button("Top-left"):
        st.session_state.gd_sx, st.session_state.gd_sy = -8.0, 6.0
    if preset[1].button("Ridge"):
        st.session_state.gd_sx, st.session_state.gd_sy = 9.0, -5.0
    if preset[2].button("Near min"):
        st.session_state.gd_sx, st.session_state.gd_sy = 2.0, 1.0
    start_x = st.slider("Start w\u2081", float(X_RANGE[0]), float(X_RANGE[1]),
                        key="gd_sx")
    start_y = st.slider("Start w\u2082", float(Y_RANGE[0]), float(Y_RANGE[1]),
                        key="gd_sy")


@st.cache_data
def run(sx, sy, eta, steps):
    return descend((sx, sy), eta, steps)


path, status = run(start_x, start_y, lr, n_steps)

# --- Visualisation --------------------------------------------------------
st.plotly_chart(descent_figure(path), width='stretch')

# --- Metrics --------------------------------------------------------------
last = path[-1]
status_label = {
    "running": "still descending",
    "converged": "converged",
    "diverged": "diverged",
}[status]
C.metric_row([
    ("Iterations", f"{len(path) - 1}"),
    ("Final loss", "diverged" if status == "diverged"
     else f"{loss(last[0], last[1]):.2f}"),
    ("Status", status_label),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("The loss is a simple bowl. We want the point where it is lowest:")
    st.latex(r"L(w_1, w_2) = 0.08\,w_1^2 + 0.5\,w_2^2")
    st.markdown("The gradient points uphill, so we step the opposite way:")
    st.latex(r"\nabla L = \big(0.16\,w_1,\; 1.0\,w_2\big)")
    st.latex(r"w \leftarrow w - \eta \,\nabla L(w)")
    st.markdown(
        "Here $\\eta$ (eta) is the **learning rate** -- how far we move on each "
        "step. The surface is steeper in $w_2$ than $w_1$, which is why one "
        "learning rate struggles to suit both directions."
    )


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Leave the learning rate at 0.30 and press **Play**. Watch the step sizes.",
     "Steps are large far from the bottom and shrink as the slope flattens. "
     "The gradient is proportional to the slope, so descent slows down near the "
     "minimum automatically -- no extra logic needed."),
    ("Drag the learning rate up to about **1.9** and play again.",
     "The path zig-zags across the steep direction while crawling along the "
     "shallow one. A single learning rate cannot suit both -- this is exactly "
     "what momentum and adaptive optimizers like Adam are built to fix."),
    ("Now push the learning rate **past 2.0**.",
     "Each step overshoots and the next is larger still. The loss explodes and "
     "the run is marked *diverged*."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "Above a learning rate of about 2.0 the method diverges: a bigger step is "
    "not always better. There is a stability ceiling set by the curvature of "
    "the loss surface."
)
