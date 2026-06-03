"""Small image and convolution helpers for the convolution module."""

import numpy as np


KERNELS = {
    "Edge detect": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float),
    "Blur": np.ones((3, 3), dtype=float) / 9.0,
    "Emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=float),
}


def sample_image(size=32):
    """Generate a tiny image with edges and smooth regions."""
    image = np.zeros((size, size), dtype=float)
    image[6:24, 8:22] = 0.45
    image[10:18, 13:27] = 0.85
    yy, xx = np.ogrid[:size, :size]
    circle = (xx - 12) ** 2 + (yy - 22) ** 2 <= 5 ** 2
    image[circle] = 1.0
    return image


def convolve2d(image, kernel):
    """Apply a 3x3 convolution kernel with edge padding."""
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if kernel.shape != (3, 3):
        raise ValueError("kernel must be 3x3")
    padded = np.pad(image, 1, mode="edge")
    output = np.zeros_like(image)
    flipped = kernel[::-1, ::-1]
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            patch = padded[row:row + 3, col:col + 3]
            output[row, col] = float(np.sum(patch * flipped))
    return output


def normalize_image(image):
    """Scale an image to [0, 1] for display."""
    image = np.asarray(image, dtype=float)
    lo = float(image.min())
    hi = float(image.max())
    if hi == lo:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)
