"""Plotly figure builders.

The descent figure draws the loss surface once (a contour) and animates only
the path trace via Plotly frames. The animation therefore runs entirely in the
browser -- no server-side loop -- which keeps it smooth on Streamlit Community
Cloud's free tier.
"""

import plotly.graph_objects as go

from lib.descent import surface_grid, X_RANGE, Y_RANGE

PATH_LINE = "#F0997B"
PATH_DOT = "#D85A30"
MIN_MARK = "#FFFFFF"


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
