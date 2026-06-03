"""Core maths for the eigenvectors and eigenvalues module."""

import numpy as np


def rotation_matrix(angle_degrees):
    """2D rotation matrix for an angle in degrees."""
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def symmetric_transform(stretch_1, stretch_2, angle_degrees):
    """Matrix with controllable eigenvalues and rotated eigenvectors."""
    rotation = rotation_matrix(angle_degrees)
    scale = np.diag([float(stretch_1), float(stretch_2)])
    return rotation @ scale @ rotation.T


def eigensystem(matrix):
    """Eigenpairs sorted from largest to smallest eigenvalue."""
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def transform_points(matrix, points):
    """Apply a 2D matrix to row-wise points."""
    return np.asarray(points, dtype=float) @ np.asarray(matrix, dtype=float).T
