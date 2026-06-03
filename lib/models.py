"""Small model helpers for classical ML and neural-network lessons."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def fit_svm(points, labels, kernel="linear", c=1.0, gamma=1.0):
    """Fit a small SVM classifier."""
    model = SVC(kernel=kernel, C=float(c), gamma=float(gamma))
    return model.fit(np.asarray(points, dtype=float), np.asarray(labels, dtype=int))


def fit_decision_tree(points, labels, max_depth=3):
    """Fit a small decision tree with a bounded depth."""
    model = DecisionTreeClassifier(max_depth=int(max_depth), random_state=0)
    return model.fit(np.asarray(points, dtype=float), np.asarray(labels, dtype=int))


def fit_random_forest(points, labels, n_estimators=15, max_depth=4):
    """Fit a deliberately small random forest for instant slider updates."""
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=0,
    )
    return model.fit(np.asarray(points, dtype=float), np.asarray(labels, dtype=int))


def fit_perceptron(points, labels, learning_rate=0.1, n_steps=80):
    """Train a single threshold neuron with the classic perceptron update."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    targets = np.where(labels == 1, 1, -1)
    weights = np.zeros(points.shape[1], dtype=float)
    bias = 0.0
    mistakes = []
    for _ in range(int(n_steps)):
        n_mistakes = 0
        for point, target in zip(points, targets):
            if target * (point @ weights + bias) <= 0:
                weights += float(learning_rate) * target * point
                bias += float(learning_rate) * target
                n_mistakes += 1
        mistakes.append(n_mistakes)
        if n_mistakes == 0:
            break
    return weights, float(bias), mistakes


def perceptron_predict(points, weights, bias):
    """Class predictions for a trained perceptron."""
    scores = np.asarray(points, dtype=float) @ np.asarray(weights, dtype=float) + float(bias)
    return (scores >= 0).astype(int)


def fit_mlp(points, labels, hidden_units=8, alpha=0.001):
    """Fit a tiny one-hidden-layer MLP."""
    model = MLPClassifier(
        hidden_layer_sizes=(int(hidden_units),),
        activation="tanh",
        alpha=float(alpha),
        max_iter=600,
        random_state=0,
    )
    return model.fit(np.asarray(points, dtype=float), np.asarray(labels, dtype=int))


def decision_grid(model, points, padding=1.0, n=90):
    """Return a 2D mesh and predicted class surface for fitted sklearn models."""
    points = np.asarray(points, dtype=float)
    x_range = (float(points[:, 0].min()) - padding, float(points[:, 0].max()) + padding)
    y_range = (float(points[:, 1].min()) - padding, float(points[:, 1].max()) + padding)
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.c_[gx.ravel(), gy.ravel()]
    if hasattr(model, "predict_proba"):
        z = model.predict_proba(grid)[:, 1]
    else:
        z = model.predict(grid)
    return xs, ys, z.reshape(gx.shape), x_range, y_range


def accuracy(model, points, labels):
    """Training accuracy for fitted sklearn classifiers."""
    return float(model.score(np.asarray(points, dtype=float), np.asarray(labels, dtype=int)))
