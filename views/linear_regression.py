"""Linear regression -- second reference module.

Same recipe as gradient_descent.py: intro -> controls -> visualisation ->
metrics -> math -> tasks -> break-it. Note how it reuses datasets.noisy_line,
the components helpers, and the regression maths in lib/regression.py -- the
page itself does no numeric work.
"""

import numpy as np
import streamlit as st

from lib import components as C, datasets
from lib.plotting import regression_figure
from lib.regression import best_fit, mse

C.intro(
    "Linear regression",
    "Find the straight line that sits closest to a cloud of points.",
)

st.session_state.setdefault("lr_m", 1.0)
st.session_state.setdefault("lr_b", 0.0)
st.session_state.setdefault("lr_seed", 0)

# --- Controls (part 1): options that change the data -----------------------
with st.sidebar:
    st.header("Controls")
    show_resid = st.checkbox("Show squared errors", value=False)
    add_outlier = st.checkbox("Add an outlier", value=False)
    if st.button("New data"):
        st.session_state.lr_seed += 1

# --- Data ------------------------------------------------------------------
x, y = datasets.noisy_line(seed=st.session_state.lr_seed)
x = np.asarray(x, dtype=float)
y = np.asarray(y, dtype=float)
if add_outlier:
    x = np.append(x, 0.0)
    y = np.append(y, 45.0)

bm, bb = best_fit(x, y)

# --- Controls (part 2): the line itself ------------------------------------
with st.sidebar:
    if st.button("Fit best line"):
        st.session_state.lr_m = float(np.clip(round(bm, 2), -5.0, 5.0))
        st.session_state.lr_b = float(np.clip(round(bb, 2), -20.0, 20.0))
    slope = st.slider("Slope (m)", -5.0, 5.0, step=0.1, key="lr_m")
    intercept = st.slider("Intercept (b)", -20.0, 20.0, step=0.5, key="lr_b")

# --- Visualisation ---------------------------------------------------------
st.plotly_chart(
    regression_figure(x, y, slope, intercept, residuals=show_resid),
    width="stretch",
)

# --- Metrics ---------------------------------------------------------------
C.metric_row([
    ("Your error (MSE)", f"{mse(x, y, slope, intercept):.1f}"),
    ("Best possible (MSE)", f"{mse(x, y, bm, bb):.1f}"),
])

# --- Maths -----------------------------------------------------------------
def _math():
    st.markdown("A line is two numbers -- a slope and an intercept:")
    st.latex(r"\hat{y} = m\,x + b")
    st.markdown("We score a line by its mean squared error, the average of the "
                "squared gaps between prediction and truth:")
    st.latex(r"\text{MSE} = \frac{1}{n}\sum_{i} (\hat{y}_i - y_i)^2")
    st.markdown("'Best fit' is the $m$ and $b$ that make this smallest. For a "
                "straight line there is an exact formula, so no iteration is "
                "needed -- unlike the surfaces gradient descent has to climb down.")


C.show_math(_math)

# --- Guided tasks ----------------------------------------------------------
C.try_this([
    ("Drag the sliders to push **your error** as low as the **best possible** "
     "number.",
     "It is hard to match by hand -- which is exactly why we want an algorithm "
     "to do it for us."),
    ("Press **Fit best line**, then tick **Show squared errors**.",
     "The grey segments are the residuals the method minimised. Their squared "
     "lengths, averaged, are the MSE."),
    ("Tick **Add an outlier**, then press **Fit best line** again.",
     "The whole line tilts toward the single far point."),
])

# --- The break-it moment ---------------------------------------------------
C.break_it(
    "One outlier visibly drags the best-fit line. Because the error is "
    "*squared*, a single large gap is punished far more than many small ones, "
    "so least squares is sensitive to outliers."
)
