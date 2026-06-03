"""Convolution -- a small kernel sliding over an image."""

import streamlit as st

from lib import components as C
from lib.convolution import KERNELS, convolve2d, normalize_image, sample_image
from lib.plotting import convolution_figure

C.intro(
    "Convolution",
    "Slide a tiny filter over an image to build a feature map.",
)

plot = st.empty()
image = sample_image(size=32)
max_pos = image.shape[0] - 3
n_positions = (max_pos + 1) * (max_pos + 1)

# --- Controls -------------------------------------------------------------
with C.controls():
    kernel_name = st.selectbox("Kernel", list(KERNELS))
    scan_step = st.slider("Kernel scan position", 0, n_positions - 1, 10 * (max_pos + 1) + 10)
    row = scan_step // (max_pos + 1)
    col = scan_step % (max_pos + 1)
    st.caption(f"Patch top-left: row {row}, column {col}")

kernel = KERNELS[kernel_name]
raw_output = convolve2d(image, kernel)
output = normalize_image(raw_output)

# --- Visualisation --------------------------------------------------------
plot.plotly_chart(convolution_figure(image, output, row, col), width="stretch")

# --- Metrics --------------------------------------------------------------
patch = image[row:row + 3, col:col + 3]
C.metric_row([
    ("Kernel", kernel_name),
    ("Patch response", f"{raw_output[row + 1, col + 1]:.2f}"),
    ("Kernel sum", f"{kernel.sum():.2f}"),
])

# --- Maths ----------------------------------------------------------------
def _math():
    st.markdown("A convolution multiplies a local patch by a kernel and sums the result:")
    st.latex(r"Y_{i,j} = \sum_m \sum_n X_{i+m,j+n}K_{m,n}")
    st.markdown("Different kernels highlight different patterns: edges, blur, sharpening, or direction changes.")
    st.write("Current 3x3 patch:")
    st.dataframe(patch, hide_index=True)
    st.write("Current kernel:")
    st.dataframe(kernel, hide_index=True)


C.show_math(_math)

# --- Guided tasks ---------------------------------------------------------
C.try_this([
    ("Choose **Edge detect**.", "Flat regions become dark while sharp boundaries light up."),
    ("Drag **Kernel scan position**.", "The highlighted square moves through the image like a sliding filter."),
    ("Switch to **Blur**.", "The feature map smooths abrupt changes by averaging neighbors."),
])

# --- The break-it moment --------------------------------------------------
C.break_it(
    "A convolution kernel only sees a small local patch. Larger meaning comes "
    "from combining many filters and layers."
)
