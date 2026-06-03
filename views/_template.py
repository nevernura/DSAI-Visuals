"""TEMPLATE -- copy this file to views/<your_module>.py to add a module.

Steps to add a module:
  1. Copy this file and rename it.
  2. Put the module's maths in lib/<your_module>.py (mirror lib/descent.py).
  3. Put any figure building in lib/plotting.py.
  4. Fill in the sections below.
  5. Register the page in Home.py under the right sidebar section.

This file is not registered in navigation (underscore prefix) and is here only
as a reference skeleton.
"""

import streamlit as st

from lib import components as C

C.intro(
    "Module title",
    "One plain-language sentence a beginner understands before any equation.",
)

# --- Controls -------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    # example_param = st.slider("Label", min, max, default)

# --- Visualisation --------------------------------------------------------
# fig = build_your_figure(...)
# st.plotly_chart(fig, use_container_width=True)

# --- Metrics (optional) ---------------------------------------------------
# C.metric_row([("Label", value), ...])

# --- Maths ----------------------------------------------------------------
def _math():
    st.latex(r"y = mx + b")
    st.markdown("Explain each symbol in one line.")


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("First thing to try.", "What the student should notice."),
    ("Second thing to try.", "What the student should notice."),
])

# --- The break-it moment --------------------------------------------------
C.break_it("The one setting where this method visibly fails, and why.")
