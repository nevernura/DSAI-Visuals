"""Linear regression -- second reference module."""

import numpy as np
import streamlit as st

from lib import components as C
from lib.data_options import REGRESSION_OPTIONS, regression_data
from lib.plotting import regression_figure
from lib.regression import best_fit, mse

C.intro(
    "Linear regression",
    "Find the straight line that sits closest to a cloud of points.",
)

st.session_state.setdefault("lr_m", 1.0)
st.session_state.setdefault("lr_b", 0.0)
st.session_state.setdefault("lr_seed", 0)

plot = st.empty()

# --- Controls -------------------------------------------------------------
with C.controls():
    data_kind = st.selectbox("Data", REGRESSION_OPTIONS)
    show_resid = st.checkbox("Show squared errors", value=False)
    if st.button("New data"):
        st.session_state.lr_seed += 1

    x, y, data_note = regression_data(data_kind, seed=st.session_state.lr_seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    st.caption(data_note)

    bm, bb = best_fit(x, y)
    if st.button("Fit best line"):
        st.session_state.lr_m = float(np.clip(round(bm, 2), -5.0, 5.0))
        st.session_state.lr_b = float(np.clip(round(bb, 2), -20.0, 50.0))
    slope = st.slider("Slope (m)", -5.0, 5.0, step=0.1, key="lr_m")
    intercept = st.slider("Intercept (b)", -20.0, 50.0, step=0.5, key="lr_b")

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(
    regression_figure(x, y, slope, intercept, residuals=show_resid),
    width="stretch",
)

# --- Metrics --------------------------------------------------------------
C.metric_row([
    ("Your error (MSE)", f"{mse(x, y, slope, intercept):.1f}"),
    ("Best possible (MSE)", f"{mse(x, y, bm, bb):.1f}"),
])

# --- Maths ----------------------------------------------------------------

def _math():
    st.markdown("A line is two numbers -- a slope and an intercept:")
    st.latex(r"\hat{y} = m\,x + b")
    st.markdown("We score a line by mean squared error:")
    st.latex(r"\text{MSE} = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2")
    st.markdown("Best fit is the slope and intercept that make this smallest.")

C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Drag the sliders to push **your error** as low as the **best possible** number.",
     "It is hard to match by hand -- that is why we want an algorithm."),
    ("Press **Fit best line**, then tick **Show squared errors**.",
     "The grey segments are the residuals the method minimized."),
    ("Switch to **Noisy line with outlier**.",
     "The whole line tilts toward the single far point."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "One outlier visibly drags the best-fit line. Because the error is squared, "
    "a single large gap is punished far more than many small ones."
)
