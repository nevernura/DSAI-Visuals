"""Maths for the linear regression module.

UI-independent, like lib/descent.py: predictions, the mean-squared-error
objective, and the closed-form least-squares solution. The page imports these
and never does numeric work itself.
"""

import numpy as np


def predict(x, slope, intercept):
    """Predicted y for each x given a line."""
    return slope * np.asarray(x, dtype=float) + intercept


def mse(x, y, slope, intercept):
    """Mean squared error of a line against the data."""
    err = predict(x, slope, intercept) - np.asarray(y, dtype=float)
    return float(np.mean(err ** 2))


def best_fit(x, y):
    """Closed-form least-squares line. Returns (slope, intercept)."""
    x = np.asarray(x, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)[0]
    return float(slope), float(intercept)
