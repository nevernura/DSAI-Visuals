"""Toy dataset generators, cached so re-runs are instant on the free tier.

Modules should pull their data from here (with a preset name and a seed for a
'regenerate' button) rather than hand-rolling arrays, so datasets stay small,
consistent, and reproducible. CSV upload can be added here later without
touching any page.
"""

import numpy as np
import streamlit as st


@st.cache_data
def blobs(n=150, centers=2, spread=1.2, seed=0):
    """Gaussian blobs for clustering / classification demos."""
    rng = np.random.default_rng(seed)
    pts, labels = [], []
    for c in range(centers):
        angle = 2 * np.pi * c / centers
        cx, cy = 5 * np.cos(angle), 5 * np.sin(angle)
        pts.append(rng.normal([cx, cy], spread, size=(n // centers, 2)))
        labels.append(np.full(n // centers, c))
    return np.vstack(pts), np.concatenate(labels)


@st.cache_data
def noisy_line(n=40, slope=1.5, intercept=2.0, noise=2.5, seed=0):
    """Points scattered around a line, for regression demos."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, size=n)
    y = slope * x + intercept + rng.normal(0, noise, size=n)
    return x, y


@st.cache_data
def two_rings(n=200, noise=0.12, seed=0):
    """Concentric rings: linearly non-separable, for SVM / MLP demos."""
    rng = np.random.default_rng(seed)
    half = n // 2
    t = rng.uniform(0, 2 * np.pi, size=half)
    inner = np.c_[np.cos(t), np.sin(t)] * 1.0 + rng.normal(0, noise, (half, 2))
    t2 = rng.uniform(0, 2 * np.pi, size=half)
    outer = np.c_[np.cos(t2), np.sin(t2)] * 2.5 + rng.normal(0, noise, (half, 2))
    pts = np.vstack([inner, outer])
    labels = np.concatenate([np.zeros(half), np.ones(half)])
    return pts, labels
