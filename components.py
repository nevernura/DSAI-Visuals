"""Shared UI components that define the look of every module.

Each module page is expected to call, in order:
    intro()        -> title + one-line plain-language description
    (the interactive visualisation + controls -- module specific)
    metric_row()   -> a few headline numbers (optional)
    show_math()    -> the equation, tucked behind an expander
    try_this()     -> 2-3 guided tasks, each with a hidden "what to notice"
    break_it()     -> the one regime where the method visibly fails

Keeping this contract identical across modules is what makes the kit feel
like one product and lets a new module be written by copying _template.py.
"""

import streamlit as st


def intro(title, tagline):
    """Module header: a title and a single jargon-free sentence."""
    st.title(title)
    st.markdown(
        f"<p style='font-size:1.1rem;color:#666;margin-top:-0.5rem'>{tagline}</p>",
        unsafe_allow_html=True,
    )


def metric_row(metrics):
    """Render a row of headline numbers. ``metrics`` is a list of (label, value)."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def show_math(render_fn):
    """Intuition first: the maths lives one click away, not in the student's face."""
    with st.expander("Show the math"):
        render_fn()


def try_this(tasks):
    """Guided self-study tasks. ``tasks`` is a list of (instruction, what_to_notice)."""
    st.subheader("Try this")
    for i, (instruction, notice) in enumerate(tasks, start=1):
        st.markdown(f"**{i}.** {instruction}")
        with st.expander("What you should notice"):
            st.markdown(notice)


def break_it(text):
    """The deliberate failure mode -- the moment understanding actually forms."""
    st.warning(text, icon=":material/warning:")
