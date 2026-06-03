"""Shared dataset choices for interactive lesson pages."""

import numpy as np
from sklearn.datasets import load_diabetes, load_iris, load_wine

from lib import datasets

CLASSIFICATION_OPTIONS = [
    "Linearly separable",
    "Overlapping",
    "Non-linear rings",
    "Real: Iris",
]

CLUSTERING_OPTIONS = [
    "Separated blobs",
    "Overlapping blobs",
    "Rings",
    "Real: Iris",
]

REGRESSION_OPTIONS = [
    "Noisy line",
    "Noisy line with outlier",
    "Real: Diabetes",
]

PCA_OPTIONS = [
    "Toy 3D cloud",
    "Real: Iris",
    "Real: Wine",
]


def _standardize(points):
    points = np.asarray(points, dtype=float)
    scale = points.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (points - points.mean(axis=0)) / scale


def classification_data(kind, seed=0, n=180):
    """Return 2D classification points, labels, and a short note."""
    if kind == "Linearly separable":
        points, labels = datasets.blobs(n=n, centers=2, spread=1.0, seed=seed)
        return points, labels.astype(int), "Two Gaussian blobs with a clear straight-line split."
    if kind == "Overlapping":
        points, labels = datasets.blobs(n=n, centers=2, spread=2.0, seed=seed)
        return points, labels.astype(int), "Two blobs with enough overlap that mistakes are unavoidable."
    if kind == "Non-linear rings":
        points, labels = datasets.two_rings(n=n, noise=0.12, seed=seed)
        return points, labels.astype(int), "Concentric rings: not separable by a single straight line."
    if kind == "Real: Iris":
        iris = load_iris()
        mask = iris.target < 2
        points = _standardize(iris.data[mask][:, [2, 3]])
        labels = iris.target[mask].astype(int)
        return points, labels, "Real Iris data: setosa vs versicolor using petal length and width."
    raise ValueError(f"Unknown classification dataset: {kind}")


def clustering_data(kind, seed=0, n=180):
    """Return 2D unlabeled points and a short note for clustering demos."""
    if kind == "Separated blobs":
        points, _ = datasets.blobs(n=n, centers=3, spread=1.05, seed=seed)
        return points, "Three compact toy clusters."
    if kind == "Overlapping blobs":
        points, _ = datasets.blobs(n=n, centers=3, spread=1.8, seed=seed)
        return points, "Three toy clusters with visible overlap."
    if kind == "Rings":
        points, _ = datasets.two_rings(n=n, noise=0.12, seed=seed)
        return points, "Rings are a bad fit for centroid-shaped clusters."
    if kind == "Real: Iris":
        iris = load_iris()
        points = _standardize(iris.data[:, [2, 3]])
        return points, "Real Iris data using petal length and width."
    raise ValueError(f"Unknown clustering dataset: {kind}")


def regression_data(kind, seed=0):
    """Return x, y, and a short note for regression demos."""
    if kind in {"Noisy line", "Noisy line with outlier"}:
        x, y = datasets.noisy_line(seed=seed)
        if kind == "Noisy line with outlier":
            x = np.append(x, 0.0)
            y = np.append(y, 45.0)
            return x, y, "A toy line with one large outlier."
        return x, y, "A small toy line with Gaussian noise."
    if kind == "Real: Diabetes":
        data = load_diabetes()
        x = _standardize(data.data[:, [2]]).ravel() * 10.0
        y = _standardize(data.target[:, None]).ravel() * 18.0 + 20.0
        return x, y, "Real diabetes data: standardized BMI vs disease progression."
    raise ValueError(f"Unknown regression dataset: {kind}")


def pca_data(kind, seed=0):
    """Return 3D points and a short note for PCA demos."""
    if kind == "Toy 3D cloud":
        return datasets.correlated_cloud_3d(seed=seed), "Synthetic 3D cloud with one dominant direction."
    if kind == "Real: Iris":
        iris = load_iris()
        return _standardize(iris.data[:, :3]), "Real Iris data using three flower measurements."
    if kind == "Real: Wine":
        wine = load_wine()
        return _standardize(wine.data[:, :3]), "Real Wine data using three chemical measurements."
    raise ValueError(f"Unknown PCA dataset: {kind}")
