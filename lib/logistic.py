"""Core maths for the logistic regression module."""

import numpy as np


def sigmoid(z):
    """Numerically stable logistic squashing function."""
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def probabilities(points, weights, bias):
    """Probability that each point belongs to class 1."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return sigmoid(points @ weights + float(bias))


def predict_classes(points, weights, bias, threshold=0.5):
    """Class predictions from logistic probabilities."""
    return (probabilities(points, weights, bias) >= threshold).astype(int)


def log_loss(points, labels, weights, bias):
    """Binary cross-entropy loss for the current classifier."""
    labels = np.asarray(labels, dtype=float)
    probs = np.clip(probabilities(points, weights, bias), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def accuracy(points, labels, weights, bias):
    """Share of examples classified correctly."""
    labels = np.asarray(labels, dtype=int)
    preds = predict_classes(points, weights, bias)
    return float(np.mean(preds == labels))


def fit(points, labels, learning_rate=0.15, n_steps=600):
    """Fit weights with batch gradient descent. Returns (weights, bias)."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array")
    if len(points) == 0:
        raise ValueError("points must contain at least one row")
    if labels.shape != (len(points),):
        raise ValueError("labels must be a 1D array with one label per point")

    weights = np.zeros(2, dtype=float)
    bias = 0.0
    for _ in range(n_steps):
        error = probabilities(points, weights, bias) - labels
        weights -= learning_rate * (points.T @ error) / len(points)
        bias -= learning_rate * float(np.mean(error))
    return weights, float(bias)
