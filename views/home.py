"""Landing page."""

import streamlit as st

st.title("Linear algebra & machine learning, made visual")
st.markdown(
    "A self-study lab for foundational data science and AI. Every page is a "
    "small lesson: an interactive picture, the equation one click away, and a "
    "few guided tasks. Pick a topic from the sidebar."
)

st.subheader("How each lesson works")
st.markdown(
    "- **See it first.** The interactive visualisation leads; the maths waits "
    "behind a *Show the math* expander.\n"
    "- **Play with it.** Drag the sliders and press play. Cause and effect are "
    "tied together on screen.\n"
    "- **Try the tasks.** Each page ends with a few prompts and a hidden "
    "*what you should notice* for each.\n"
    "- **Break it.** Every topic has one setting where the method visibly "
    "fails -- that is where the real understanding forms."
)

st.subheader("The roadmap")
st.markdown(
    "**Phase 1 - the core path:** gradient descent, linear regression, "
    "and logistic regression are live.\n\n"
    "**Phase 2 - expansion:** eigenvectors, PCA, SVM, decision tree, "
    "and k-means are live.\n\n"
    "**Phase 3 - deep learning and ensembles:** random forest, perceptron, "
    "MLP, and convolution are live."
)

st.info(
    "This build ships the shared template, the machine-learning sequence, "
    "and the first deep-learning and ensemble lessons.",
    icon=":material/build:",
)
