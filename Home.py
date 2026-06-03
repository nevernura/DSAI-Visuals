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

eigenvectors = st.Page(
    VIEWS / "eigenvectors.py", title="Eigenvectors and eigenvalues",
    icon=":material/open_in_full:",
)

pca = st.Page(
    VIEWS / "pca.py", title="PCA",
    icon=":material/view_in_ar:",
)

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


svm = st.Page(
    VIEWS / "svm.py", title="SVM",
    icon=":material/commit:",
)

decision_tree = st.Page(
    VIEWS / "decision_tree.py", title="Decision tree",
    icon=":material/account_tree:",
)

k_means = st.Page(
    VIEWS / "k_means.py", title="K-means",
    icon=":material/hub:",
)

random_forest = st.Page(
    VIEWS / "random_forest.py", title="Random forest",
    icon=":material/forest:",
)

perceptron = st.Page(
    VIEWS / "perceptron.py", title="Perceptron",
    icon=":material/radio_button_checked:",
)

mlp = st.Page(
    VIEWS / "mlp.py", title="MLP",
    icon=":material/schema:",
)

convolution = st.Page(
    VIEWS / "convolution.py", title="Convolution",
    icon=":material/filter_center_focus:",
)

# As Phase 1+ modules are written, add them under the matching section, e.g.
#   "Linear algebra": [vectors, matrix_multiply, separability],
#   "Deep learning":  [perceptron, mlp, convolution],
navigation = st.navigation({
    "Start here": [home],
    "Linear algebra": [eigenvectors, pca],
    "Machine learning": [
        gradient_descent, linear_regression, logistic_regression,
        svm, decision_tree, k_means,
    ],
    "Deep learning and ensembles": [random_forest, perceptron, mlp, convolution],
})

navigation.run()
