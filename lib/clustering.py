"""Core maths for the k-means module."""

import numpy as np


def initial_centroids(points, k=3, bad=False):
    """Choose deterministic starting centroids for a repeatable lesson."""
    points = np.asarray(points, dtype=float)
    if bad:
        order = np.argsort(points[:, 0])
        idx = order[:k]
    else:
        quantiles = np.linspace(0.15, 0.85, k)
        idx = [int(q * (len(points) - 1)) for q in quantiles]
        idx = np.argsort(points[:, 0])[idx]
    return points[idx].copy()


def assign(points, centroids):
    """Assign each point to the nearest centroid."""
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    distances = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


def recenter(points, labels, centroids):
    """Move each centroid to the mean of its assigned points."""
    points = np.asarray(points, dtype=float)
    next_centroids = np.asarray(centroids, dtype=float).copy()
    for idx in range(len(next_centroids)):
        mask = labels == idx
        if np.any(mask):
            next_centroids[idx] = points[mask].mean(axis=0)
    return next_centroids


def run_kmeans(points, k=3, bad_init=False, max_steps=8):
    """Precompute assign-then-recenter states for a scrubber animation."""
    centroids = initial_centroids(points, k=k, bad=bad_init)
    history = []
    labels = assign(points, centroids)
    history.append(("assign", centroids.copy(), labels.copy()))
    for _ in range(max_steps):
        new_centroids = recenter(points, labels, centroids)
        history.append(("recenter", new_centroids.copy(), labels.copy()))
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
        labels = assign(points, centroids)
        history.append(("assign", centroids.copy(), labels.copy()))
    return history


def inertia(points, labels, centroids):
    """Sum of squared distances to assigned centroids."""
    points = np.asarray(points, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    return float(np.sum((points - centroids[labels]) ** 2))
