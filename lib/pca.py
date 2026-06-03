"""Core maths for the PCA module."""

import numpy as np


def fit_pca(points):
    """Return mean, principal components, variances, and explained variance ratios."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (n, 3) array")
    if len(points) < 2:
        raise ValueError("PCA needs at least two points")

    mean = points.mean(axis=0)
    centered = points - mean
    covariance = np.cov(centered, rowvar=False)
    variances, components = np.linalg.eigh(covariance)
    order = np.argsort(variances)[::-1]
    variances = variances[order]
    components = components[:, order].T
    explained = variances / variances.sum()
    scores = centered @ components.T
    return mean, components, variances, explained, scores


def reconstruct(scores, components, mean, n_components):
    """Project points onto the first n principal components and lift back to 3D."""
    scores = np.asarray(scores, dtype=float)
    components = np.asarray(components, dtype=float)
    mean = np.asarray(mean, dtype=float)
    kept_scores = scores[:, :n_components]
    kept_components = components[:n_components]
    return kept_scores @ kept_components + mean
