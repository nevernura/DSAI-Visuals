# lib/regression.py
import numpy as np

def predict(x, slope, intercept):
    return slope * x + intercept

def mse(x, y, slope, intercept):
    return float(np.mean((predict(x, slope, intercept) - y) ** 2))

def best_fit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)

def regression_figure(x, y, slope, intercept):
    import numpy as np
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="markers", marker=dict(color=PATH_DOT), showlegend=False)
    ends = np.array([x.min(), x.max()])
    fig.add_scatter(x=ends, y=slope * ends + intercept, mode="lines",
                    line=dict(color=PATH_LINE, width=3), showlegend=False)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

# views/linear_regression.py
import streamlit as st
from lib import components as C, datasets
from lib.regression import mse, best_fit
from lib.plotting import regression_figure

C.intro("Linear regression", "Find the straight line that sits closest to a cloud of points.")

with st.sidebar:
    st.header("Controls")
    slope = st.slider("Slope (m)", -5.0, 5.0, 1.0, 0.1)
    intercept = st.slider("Intercept (b)", -10.0, 10.0, 0.0, 0.5)

x, y = datasets.noisy_line()
st.plotly_chart(regression_figure(x, y, slope, intercept), width="stretch")

bm, bb = best_fit(x, y)
C.metric_row([("Your line's error", f"{mse(x, y, slope, intercept):.1f}"),
              ("Best possible error", f"{mse(x, y, bm, bb):.1f}")])

def _math():
    st.latex(r"\hat{y} = m x + b")
    st.latex(r"\text{MSE} = \tfrac{1}{n}\sum_i (\hat{y}_i - y_i)^2")
    st.markdown("We pick the $m$ and $b$ that make the average squared error smallest.")
C.show_math(_math)

C.try_this([
    ("Adjust the sliders to push your error as low as the 'best possible' number.",
     "It's hard to hit by hand — which is exactly why we need an algorithm to do it."),
])
C.break_it("Try adding one far-away point: least squares chases it, because squaring "
           "punishes large errors hardest. Linear regression is sensitive to outliers.")

linear_regression = st.Page(VIEWS / "linear_regression.py",
                            title="Linear regression", icon=":material/show_chart:")

navigation = st.navigation({
    "Start here": [home],
    "Machine learning": [gradient_descent, linear_regression],
})
