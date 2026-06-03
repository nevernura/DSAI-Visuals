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
    "**Phase 1 - the core path (live now in part):** vectors & the dot product "
    "\u2192 matrix multiplication \u2192 separability \u2192 gradient descent "
    "\u2192 linear regression \u2192 logistic regression.\n\n"
    "**Phase 2 - expansion:** eigenvectors, PCA, SVM, decision trees, k-means.\n\n"
    "**Phase 3 - deep learning:** random forest, perceptron, MLP, convolution."
)

st.info(
    "This build ships the shared template and the gradient descent module as "
    "the reference. New modules are added by copying that one page.",
    icon=":material/build:",
)
