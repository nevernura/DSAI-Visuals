"""Core maths for the linear algebra intro modules."""

import numpy as np


def vector_from_polar(length, angle_degrees):
    """2D vector from length and angle in degrees."""
    theta = np.deg2rad(angle_degrees)
    return float(length) * np.array([np.cos(theta), np.sin(theta)], dtype=float)


def dot(a, b):
    """Dot product of two vectors."""
    return float(np.dot(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))


def vector_norm(v):
    """Euclidean length of a vector."""
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def angle_between(a, b):
    """Smallest angle between two vectors, in degrees."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = vector_norm(a) * vector_norm(b)
    if denom == 0.0:
        return 0.0
    cos_theta = np.clip(dot(a, b) / denom, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_theta)))


def project_onto(vector, direction):
    """Projection of vector onto direction."""
    vector = np.asarray(vector, dtype=float)
    direction = np.asarray(direction, dtype=float)
    denom = dot(direction, direction)
    if denom == 0.0:
        return np.zeros_like(vector)
    return dot(vector, direction) / denom * direction


def rotation_matrix(angle_degrees):
    """2D rotation matrix."""
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def scale_matrix(scale_x, scale_y):
    """2D scale matrix."""
    return np.array([[float(scale_x), 0.0], [0.0, float(scale_y)]], dtype=float)


def shear_matrix(shear_x):
    """Horizontal shear matrix."""
    return np.array([[1.0, float(shear_x)], [0.0, 1.0]], dtype=float)


def transform_matrix(scale_x=1.0, scale_y=1.0, shear_x=0.0, angle_degrees=0.0):
    """Compose scale, then shear, then rotation."""
    return rotation_matrix(angle_degrees) @ shear_matrix(shear_x) @ scale_matrix(scale_x, scale_y)


def named_matrix(name):
    """Small named transform for matrix multiplication demos."""
    if name == "Rotate 45 degrees":
        return rotation_matrix(45)
    if name == "Scale x":
        return scale_matrix(1.8, 0.8)
    if name == "Shear":
        return shear_matrix(0.9)
    if name == "Reflect x":
        return scale_matrix(1.0, -1.0)
    raise ValueError(f"Unknown matrix preset: {name}")
