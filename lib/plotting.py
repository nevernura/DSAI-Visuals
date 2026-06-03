"""Plotly figure builders.

The descent figure draws the loss surface once (a contour) and animates only
the path trace via Plotly frames. The animation therefore runs entirely in the
browser -- no server-side loop -- which keeps it smooth on Streamlit Community
Cloud's free tier. Regression/classification figures are static and reuse the
same visual language.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def eigen_figure(matrix, values, vectors, show_circle=True):
    """Show a 2D matrix transform and the directions it leaves unchanged."""
    matrix = np.asarray(matrix, dtype=float)
    values = np.asarray(values, dtype=float)
    vectors = np.asarray(vectors, dtype=float)

    fig = go.Figure()

    if show_circle:
        theta = np.linspace(0.0, 2.0 * np.pi, 160)
        circle = np.c_[np.cos(theta), np.sin(theta)]
        transformed = circle @ matrix.T
        fig.add_scatter(
            x=circle[:, 0], y=circle[:, 1], mode="lines",
            line=dict(color="rgba(255,255,255,0.35)", width=1),
            name="unit circle", hoverinfo="skip",
        )
        fig.add_scatter(
            x=transformed[:, 0], y=transformed[:, 1], mode="lines",
            line=dict(color=PATH_LINE, width=3),
            name="transformed circle", hoverinfo="skip",
        )

    basis = np.array([[1.0, 0.0], [0.0, 1.0]])
    transformed_basis = basis @ matrix.T
    for start, end, color, name in [
        (np.zeros(2), transformed_basis[0], "#4C78A8", "A e1"),
        (np.zeros(2), transformed_basis[1], "#72B7B2", "A e2"),
    ]:
        fig.add_scatter(
            x=[start[0], end[0]], y=[start[1], end[1]], mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            name=name, hoverinfo="skip",
        )

    for i in range(2):
        vector = vectors[:, i]
        scaled = values[i] * vector
        fig.add_scatter(
            x=[-scaled[0], scaled[0]], y=[-scaled[1], scaled[1]],
            mode="lines",
            line=dict(color=CLASS_1 if i == 0 else CLASS_0, width=4, dash="dash"),
            name=f"eigenvector {i + 1}", hoverinfo="skip",
        )

    limit = max(3.0, float(np.max(np.abs(matrix))) * 2.5)
    fig.update_layout(
        height=500, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="x", range=[-limit, limit], zeroline=True, scaleanchor="y"),
        yaxis=dict(title="y", range=[-limit, limit], zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def pca_figure(points, mean, components, scores, n_components=2, show_shadow=True):
    """3D PCA cloud with principal axes and an optional projection shadow."""
    points = np.asarray(points, dtype=float)
    mean = np.asarray(mean, dtype=float)
    components = np.asarray(components, dtype=float)
    scores = np.asarray(scores, dtype=float)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("3D cloud and principal directions", "PCA shadow"),
        column_widths=[0.58, 0.42],
    )

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=3, color=PATH_DOT, opacity=0.7),
            name="points", hoverinfo="skip",
        ),
        row=1, col=1,
    )

    if show_shadow:
        kept = max(1, min(int(n_components), 2))
        projected = scores[:, :kept] @ components[:kept] + mean
        fig.add_trace(
            go.Scatter3d(
                x=projected[:, 0], y=projected[:, 1], z=projected[:, 2],
                mode="markers",
                marker=dict(size=2, color="rgba(136,135,128,0.45)"),
                name="projection", hoverinfo="skip",
            ),
            row=1, col=1,
        )
        for point, shadow in zip(points[::12], projected[::12]):
            fig.add_trace(
                go.Scatter3d(
                    x=[point[0], shadow[0]], y=[point[1], shadow[1]],
                    z=[point[2], shadow[2]], mode="lines",
                    line=dict(color="rgba(136,135,128,0.25)", width=1),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1,
            )

    axis_colors = [CLASS_1, CLASS_0, "#72B7B2"]
    for i, component in enumerate(components):
        length = float(np.std(scores[:, i]) * 2.5)
        start = mean - component * length
        end = mean + component * length
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                mode="lines",
                line=dict(color=axis_colors[i], width=6),
                name=f"PC{i + 1}", hoverinfo="skip",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=scores[:, 0],
            y=scores[:, 1] if n_components == 2 else np.zeros(len(scores)),
            mode="markers",
            marker=dict(size=6, color=PATH_DOT, opacity=0.75),
            name="shadow", hoverinfo="skip",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="PC1 score", zeroline=False, row=1, col=2)
    fig.update_yaxes(title_text="PC2 score" if n_components == 2 else "collapsed",
                     zeroline=False, row=1, col=2)
    fig.update_layout(
        height=540, margin=dict(l=10, r=10, t=45, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_scenes(
        xaxis_title="x", yaxis_title="y", zaxis_title="z",
        xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
        zaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
    )
    return fig

