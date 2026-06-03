"""Entry point. Run with:  streamlit run Home.py

Navigation is grouped into sections so the sidebar reads like a curriculum:
Linear Algebra -> Machine Learning -> Deep Learning. To add a module, create
its page in views/ and add an st.Page line to the right section below.

Page paths are anchored to this file's own location so they resolve the same
way locally and on Streamlit Community Cloud, regardless of working directory.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="DS & AI Visual Lab",
    page_icon=":material/insights:",
    layout="wide",
)

VIEWS = Path(__file__).parent / "views"

home = st.Page(VIEWS / "home.py", title="Welcome", icon=":material/home:",
               default=True)

gradient_descent = st.Page(
    VIEWS / "gradient_descent.py", title="Gradient descent",
    icon=":material/trending_down:",
)

linear_regression = st.Page(
    VIEWS / "linear_regression.py", title="Linear regression",
    icon=":material/show_chart:",
)

logistic_regression = st.Page(
    VIEWS / "logistic_regression.py", title="Logistic regression",
    icon=":material/call_split:",
)

# As Phase 1+ modules are written, add them under the matching section, e.g.
#   "Linear algebra": [vectors, matrix_multiply, separability],
#   "Deep learning":  [perceptron, mlp, convolution],
navigation = st.navigation({
    "Start here": [home],
    "Machine learning": [gradient_descent, linear_regression, logistic_regression],
})

navigation.run()
