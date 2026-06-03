"""Plotly figure builders.

The descent figure draws the loss surface once (a contour) and animates only
the path trace via Plotly frames. The animation therefore runs entirely in the
browser -- no server-side loop -- which keeps it smooth on Streamlit Community
Cloud's free tier. Regression/classification figures are static and reuse the
same visual language.
"""

import numpy as np
import plotly.graph_objects as go

from lib.descent import surface_grid, X_RANGE, Y_RANGE
from lib.logistic import probabilities

PATH_LINE = "#F0997B"
PATH_DOT = "#D85A30"
MIN_MARK = "#FFFFFF"
RESID = "rgba(136,135,128,0.55)"
CLASS_0 = "#4C78A8"
CLASS_1 = "#D85A30"


def descent_figure(path):
    """Build a contour + animated descent path figure from an (n, 2) path."""
    xs, ys, z = surface_grid()

    fig = go.Figure()
    # Trace 0: the static loss surface.
    fig.add_contour(
        x=xs, y=ys, z=z, colorscale="Viridis", showscale=False, opacity=0.95,
        contours=dict(showlabels=False), hoverinfo="skip",
    )
    # Trace 1: the true minimum.
    fig.add_scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(symbol="x", size=13, color=MIN_MARK, line=dict(width=2)),
        name="minimum", hoverinfo="skip", showlegend=False,
    )
    # Trace 2: the full path (this is the trace the frames animate).
    fig.add_scatter(
        x=path[:, 0], y=path[:, 1], mode="lines+markers",
        line=dict(color=PATH_LINE, width=2.5),
        marker=dict(size=6, color=PATH_DOT),
        name="path", hoverinfo="skip", showlegend=False,
    )

    n = len(path)
    fig.frames = [
        go.Frame(
            data=[go.Scatter(
                x=path[:k, 0], y=path[:k, 1], mode="lines+markers",
                line=dict(color=PATH_LINE, width=2.5),
                marker=dict(size=6, color=PATH_DOT),
            )],
            traces=[2], name=str(k),
        )
        for k in range(1, n + 1)
    ]

    play = dict(
        type="buttons", showactive=False, x=0.0, y=1.12, xanchor="left",
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=120, redraw=False),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
        ],
    )
    slider = dict(
        x=0.0, len=1.0, currentvalue=dict(prefix="Step "),
        steps=[dict(method="animate", label=str(k),
                    args=[[str(k)], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=False))])
               for k in range(1, n + 1)],
    )

    fig.update_layout(
        updatemenus=[play], sliders=[slider],
        xaxis=dict(title="Parameter w\u2081", range=list(X_RANGE), zeroline=False),
        yaxis=dict(title="Parameter w\u2082", range=list(Y_RANGE), zeroline=False),
        height=520, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def regression_figure(x, y, slope, intercept, residuals=False):
    """Scatter of (x, y) with a fitted line; optional residual segments."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig = go.Figure()

    if residuals:
        rx, ry = [], []
        for xi, yi in zip(x, y):
            rx += [xi, xi, None]
            ry += [yi, slope * xi + intercept, None]
        fig.add_scatter(
            x=rx, y=ry, mode="lines",
            line=dict(color=RESID, width=1),
            hoverinfo="skip", showlegend=False,
        )

    fig.add_scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=8, color=PATH_DOT),
        hoverinfo="skip", showlegend=False,
    )

    ends = np.array([float(x.min()), float(x.max())])
    fig.add_scatter(
        x=ends, y=slope * ends + intercept, mode="lines",
        line=dict(color=PATH_LINE, width=3),
        hoverinfo="skip", showlegend=False,
    )

    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="x", zeroline=False),
        yaxis=dict(title="y", zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def logistic_figure(points, labels, weights, bias, show_probability=True):
    """Scatter of labelled points with a logistic decision boundary."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=float)

    fig = go.Figure()
    x_range = (float(points[:, 0].min()) - 1.0, float(points[:, 0].max()) + 1.0)
    y_range = (float(points[:, 1].min()) - 1.0, float(points[:, 1].max()) + 1.0)

    if show_probability:
        xs = np.linspace(*x_range, 80)
        ys = np.linspace(*y_range, 80)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.c_[gx.ravel(), gy.ravel()]
        z = probabilities(grid, weights, bias).reshape(gx.shape)
        fig.add_contour(
            x=xs, y=ys, z=z, colorscale="RdBu", reversescale=True,
            contours=dict(start=0.0, end=1.0, size=0.1, showlabels=False),
            opacity=0.35, showscale=False, hoverinfo="skip",
        )

    mask = labels == 1
    fig.add_scatter(
        x=points[~mask, 0], y=points[~mask, 1], mode="markers",
        marker=dict(size=8, color=CLASS_0, line=dict(width=1, color="white")),
        name="class 0", hoverinfo="skip",
    )
    fig.add_scatter(
        x=points[mask, 0], y=points[mask, 1], mode="markers",
        marker=dict(size=8, color=CLASS_1, line=dict(width=1, color="white")),
        name="class 1", hoverinfo="skip",
    )

    w1, w2 = weights
    if abs(w2) > 1e-9:
        xs = np.array(x_range)
        ys = -(w1 * xs + bias) / w2
        fig.add_scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=PATH_LINE, width=3),
            name="p = 0.5", hoverinfo="skip",
        )
    elif abs(w1) > 1e-9:
        x0 = -bias / w1
        fig.add_scatter(
            x=[x0, x0], y=list(y_range), mode="lines",
            line=dict(color=PATH_LINE, width=3),
            name="p = 0.5", hoverinfo="skip",
        )

    fig.update_layout(
        height=500, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="feature 1", range=list(x_range), zeroline=False),
        yaxis=dict(title="feature 2", range=list(y_range), zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
