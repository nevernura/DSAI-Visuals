"""Entry point. Run with:  streamlit run Home.py

Navigation is grouped into sections so the sidebar reads like a curriculum:
Linear Algebra -> Machine Learning -> Deep Learning. To add a module, create
its page in views/ and add an st.Page line to the right section below.
"""

import streamlit as st

st.set_page_config(
    page_title="DS & AI Visual Lab",
    page_icon=":material/insights:",
    layout="wide",
)

home = st.Page("views/home.py", title="Welcome", icon=":material/home:",
               default=True)

gradient_descent = st.Page(
    "views/gradient_descent.py", title="Gradient descent",
    icon=":material/trending_down:",
)

# As Phase 1+ modules are written, add them under the matching section, e.g.
#   "Linear algebra": [vectors, matrix_multiply, separability],
#   "Deep learning":  [perceptron, mlp, convolution],
navigation = st.navigation({
    "Start here": [home],
    "Machine learning": [gradient_descent],
})

navigation.run()
