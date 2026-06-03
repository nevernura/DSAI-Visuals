"""Core maths for the gradient descent module.

Everything UI-independent lives here so it can be unit-tested and reused.
To build a new module, this is the file you replace: keep the same shape
(a function that turns control values into something plottable) and the
page and plotting helpers barely change.
"""

import numpy as np

# A simple bowl-shaped loss: steeper along w2 than w1, so a single
# learning rate cannot suit both directions at once. That asymmetry is
# what produces the zig-zag at high learning rates.
AX, AY = 0.08, 0.5

X_RANGE = (-10.0, 10.0)
Y_RANGE = (-7.0, 7.0)


def loss(w1, w2):
    """Loss surface value at (w1, w2). Works on scalars or numpy arrays."""
    return AX * w1 ** 2 + AY * w2 ** 2


def gradient(w1, w2):
    """Gradient of the loss at a single point, returned as a 2-vector."""
    return np.array([2 * AX * w1, 2 * AY * w2], dtype=float)


def descend(start, learning_rate, n_steps=40, tol=1e-3):
    """Run gradient descent from ``start`` for up to ``n_steps`` iterations.

    Returns (path, status) where path is an (n, 2) array of visited points
    and status is one of "converged", "running", or "diverged".
    """
    pts = [np.array(start, dtype=float)]
    status = "running"
    for _ in range(n_steps):
        cur = pts[-1]
        nxt = cur - learning_rate * gradient(cur[0], cur[1])
        if not np.all(np.isfinite(nxt)) or np.max(np.abs(nxt)) > 1e4:
            status = "diverged"
            break
        pts.append(nxt)
        if np.linalg.norm(nxt - cur) < tol:
            status = "converged"
            break
    return np.array(pts), status


def surface_grid(n=80):
    """Sample the loss surface on a grid for contour plotting."""
    xs = np.linspace(*X_RANGE, n)
    ys = np.linspace(*Y_RANGE, n)
    gx, gy = np.meshgrid(xs, ys)
    z = loss(gx, gy)
    return xs, ys, z
